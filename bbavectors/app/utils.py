import os
import torch
import numpy as np
from bbavectors.configs import cfg, load_config
from bbavectors.models import ctrbox_net
from bbavectors.decoder import DecDecoder


def decode_prediction(orig_shape, predictions, cfg):
    down_ratio = 4
    predictions = predictions[0, :, :]
    input_h, input_w, _ = cfg.INPUT_SHAPE
    h, w, _ = orig_shape

    pts0 = {cat: [] for cat in cfg.CATEGORIES}
    scores0 = {cat: [] for cat in cfg.CATEGORIES}
    for pred in predictions:
        cen_pt = np.asarray([pred[0], pred[1]], np.float32)
        tt = np.asarray([pred[2], pred[3]], np.float32)
        rr = np.asarray([pred[4], pred[5]], np.float32)
        bb = np.asarray([pred[6], pred[7]], np.float32)
        ll = np.asarray([pred[8], pred[9]], np.float32)
        tl = tt + ll - cen_pt
        bl = bb + ll - cen_pt
        tr = tt + rr - cen_pt
        br = bb + rr - cen_pt
        score = pred[10]
        clse = pred[11]
        pts = np.asarray([tr, br, bl, tl], np.float32)
        pts[:, 0] = pts[:, 0] * down_ratio / input_w * w
        pts[:, 1] = pts[:, 1] * down_ratio / input_h * h
        pts0[cfg.CATEGORIES[int(clse)]].append(pts)
        scores0[cfg.CATEGORIES[int(clse)]].append(score)
    return pts0, scores0


def load_model(model_dir, cfg, device):
    weights_path = os.path.join(model_dir, 'model_best.pth')

    num_classes = len(cfg.CATEGORIES)
    heads = {
        'hm': num_classes,
        'wh': 10,
        'reg': 2,
        'cls_theta': 1
    }
    model = ctrbox_net.CTRBOX(
        heads=heads,
        pretrained=True,
        down_ratio=4,
        final_kernel=1,
        head_conv=256
    )
    decoder = DecDecoder(
        K=cfg.MAX_OBJECTS,
        conf_thresh=cfg.CONF_THRESH,
        num_classes=num_classes
    )
    checkpoint = torch.load(
        weights_path, map_location=lambda storage, loc: storage)

    print('loaded weights from {}, epoch {}'.format(
        weights_path, checkpoint['epoch']))

    state_dict_ = checkpoint['model_state_dict']
    model.load_state_dict(state_dict_, strict=False)
    model = model.to(device)
    model.eval()
    return model, decoder
