import torch
import os
import func_utils
from datasets.dataset import Dataset
from bbavectors import WORK_DIR


class EvalModule(object):
    def __init__(self, num_classes, model, decoder):
        torch.manual_seed(317)
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.model = model
        self.decoder = decoder

    def load_model(self, model, resume):
        checkpoint = torch.load(
            resume, map_location=lambda storage, loc: storage)
        print('loaded weights from {}, epoch {}'.format(
            resume, checkpoint['epoch']))
        state_dict_ = checkpoint['model_state_dict']
        model.load_state_dict(state_dict_, strict=False)
        return model

    def evaluation(self, args, down_ratio):
        save_path = os.path.join(WORK_DIR, 'weights')
        self.model = self.load_model(
            self.model, os.path.join(save_path, args.resume)
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        result_path = os.path.join(WORK_DIR, 'result')
        if not os.path.exists(result_path):
            os.mkdir(result_path)

        dsets = Dataset(data_dir=args.data_dir,
                        phase='test',
                        num_classes=args.num_classes,
                        input_h=args.input_h,
                        input_w=args.input_w,
                        down_ratio=down_ratio)

        func_utils.write_results(args,
                                 self.model,
                                 dsets,
                                 down_ratio,
                                 self.device,
                                 self.decoder,
                                 result_path,
                                 print_ps=True)

        if args.dataset == 'dota' or args.dataset == 'custom':
            merge_path = os.path.join(WORK_DIR, 'merge')
            if not os.path.exists(merge_path):
                os.mkdir(merge_path)
            dsets.merge_crop_image_results(result_path, merge_path)
            return None
        else:
            ap = dsets.dec_evaluation(result_path)
            return ap
