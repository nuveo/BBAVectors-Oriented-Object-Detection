import os
import cv2
import numpy as np

import torch
import torch.nn as nn
import loss
from tqdm import tqdm
from . import func_utils
from datasets.dataset import Dataset
from bbavectors import WORK_DIR


def collater(data):
    out_data_dict = {}
    for name in data[0]:
        out_data_dict[name] = []
    for sample in data:
        for name in sample:
            out_data_dict[name].append(torch.from_numpy(sample[name]))
    for name in out_data_dict:
        out_data_dict[name] = torch.stack(out_data_dict[name], dim=0)
    return out_data_dict


class TrainModule(object):
    def __init__(self, num_classes, model, decoder, down_ratio):
        torch.manual_seed(317)
        self.dataset_phase = ['train']  # hrsc = ['train', 'test']
        self.num_classes = num_classes
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.decoder = decoder
        self.down_ratio = down_ratio

    def save_model(self, path, epoch, model, loss, optimizer=None):
        if isinstance(model, torch.nn.DataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()

        if optimizer is None:
            optimizer_state_dict = None
        else:
            optimizer_state_dict = optimizer.state_dict()

        torch.save({
            'epoch': epoch,
            'model_state_dict': state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'loss': loss
        }, path)

    def load_model(self, model, optimizer, resume, strict=True):
        checkpoint = torch.load(
            resume, map_location=lambda storage, loc: storage)
        print('loaded weights from {}, epoch {}'.format(
            resume, checkpoint['epoch']))
        state_dict_ = checkpoint['model_state_dict']
        state_dict = {}
        for k in state_dict_:
            if k.startswith('module') and not k.startswith('module_list'):
                state_dict[k[7:]] = state_dict_[k]
            else:
                state_dict[k] = state_dict_[k]
        model_state_dict = model.state_dict()
        if not strict:
            for k in state_dict:
                if k in model_state_dict:
                    if state_dict[k].shape != model_state_dict[k].shape:
                        print('Skip loading parameter {}, required shape{}, '
                              'loaded shape{}.'.format(k, model_state_dict[k].shape, state_dict[k].shape))
                        state_dict[k] = model_state_dict[k]
                else:
                    print('Drop parameter {}.'.format(k))
            for k in model_state_dict:
                if not (k in state_dict):
                    print('No param {}.'.format(k))
                    state_dict[k] = model_state_dict[k]
        model.load_state_dict(state_dict, strict=False)
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        epoch = checkpoint['epoch']
        loss = checkpoint.get('loss', np.inf)
        return model, optimizer, epoch, loss

    def train_network(self, args):

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), args.init_lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.96, last_epoch=-1)
        save_path = os.path.join(WORK_DIR, 'weights')
        start_epoch = 1
        best_loss = np.inf

        # add resume part for continuing training when break previously, 10-16-2020
        if args.resume_train:
            self.model, self.optimizer, start_epoch, best_loss = self.load_model(self.model,
                                                                                 self.optimizer,
                                                                                 args.resume_train,
                                                                                 strict=True)
        # end

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if args.ngpus > 1:
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
                self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

        criterion = loss.LossAll()
        print('Setting up data...')

        dsets = {x: Dataset(data_dir=args.data_dir,
                            phase=x,
                            num_classes=args.num_classes,
                            input_h=args.input_h,
                            input_w=args.input_w,
                            down_ratio=self.down_ratio)
                 for x in self.dataset_phase}

        dsets_loader = {}
        dsets_loader['train'] = torch.utils.data.DataLoader(dsets['train'],
                                                            batch_size=args.batch_size,
                                                            shuffle=True,
                                                            num_workers=args.num_workers,
                                                            pin_memory=True,
                                                            drop_last=True,
                                                            collate_fn=collater)

        print('Starting training...')
        train_loss = []
        ap_list = []
        for epoch in range(start_epoch, args.num_epoch+1):
            print('-'*10)
            print('Epoch: {}/{} '.format(epoch, args.num_epoch))
            epoch_loss = self.run_epoch(phase='train',
                                        data_loader=dsets_loader['train'],
                                        criterion=criterion)
            train_loss.append(epoch_loss)
            self.scheduler.step(epoch)

            np.savetxt(os.path.join(save_path, 'train_loss.txt'),
                       train_loss, fmt='%.6f')

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                self.save_model(os.path.join(save_path, 'model_best.pth'),
                                epoch,
                                self.model,
                                best_loss)

            if 'test' in self.dataset_phase and epoch % 5 == 0:
                mAP = self.dec_eval(args, dsets['test'])
                ap_list.append(mAP)
                np.savetxt(os.path.join(save_path, 'ap_list.txt'),
                           ap_list, fmt='%.6f')

            self.save_model(os.path.join(save_path, 'model_last.pth'),
                            epoch,
                            self.model,
                            best_loss,
                            self.optimizer)

    def run_epoch(self, phase, data_loader, criterion):
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()
        running_loss = 0.
        for data_dict in tqdm(data_loader):
            for name in data_dict:
                data_dict[name] = data_dict[name].to(
                    device=self.device, non_blocking=True)
            if phase == 'train':
                self.optimizer.zero_grad()
                with torch.enable_grad():
                    pr_decs = self.model(data_dict['input'])
                    loss = criterion(pr_decs, data_dict)
                    loss.backward()
                    self.optimizer.step()
            else:
                with torch.no_grad():
                    pr_decs = self.model(data_dict['input'])
                    loss = criterion(pr_decs, data_dict)

            running_loss += loss.item()
        epoch_loss = running_loss / len(data_loader)
        print('{} loss: {}'.format(phase, epoch_loss))
        return epoch_loss

    def dec_eval(self, args, dsets):
        result_path = os.path.join(WORK_DIR, 'result')
        if not os.path.exists(result_path):
            os.mkdir(result_path)

        self.model.eval()
        func_utils.write_results(args,
                                 self.model, dsets,
                                 self.down_ratio,
                                 self.device,
                                 self.decoder,
                                 result_path)
        ap = dsets.dec_evaluation(result_path)
        return ap
