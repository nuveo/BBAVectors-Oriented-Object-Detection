import os
import cv2
import json
import glob
import torch
import shutil
import bisect
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from bbavectors import WORK_DIR, TEMP_DIR
from bbavectors.models import ctrbox_net
from bbavectors.decoder import DecDecoder
from DOTA_devkit.dota_utils import Task2groundtruth_poly
from DOTA_devkit.SplitOnlyImage import splitbase
from DOTA_devkit.ResultMerge_multi_process import mergebypoly


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
        pretrained=False,
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


def clear_temp_folder():
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)


def convert_altitude_to_cmpx(alt, cfg):
    cfg_res = dict(cfg.RESOLUTION)
    if not cfg_res:
        return 1.0

    if alt in cfg_res:
        return cfg_res[alt]

    res = sorted(list(cfg_res.items()))
    idx = bisect.bisect(res, (alt, 0))
    if idx == 0 or idx == len(res):
        return res[max(idx-1, 0)]

    # interpolate
    min_alt, min_cmpx = res[idx-1]
    max_alt, max_cmpx = res[idx]
    cmpx = max_cmpx * ((alt - min_alt) / max_alt) + min_cmpx
    return cmpx


def generate_splits(image, altitude, cfg):
    clear_temp_folder()
    IMAGE_DIR = os.path.join(TEMP_DIR, 'image')
    SPLIT_DIR = os.path.join(TEMP_DIR, 'split')
    IMAGE_PATH = os.path.join(IMAGE_DIR, 'test_image.jpg')
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(SPLIT_DIR, exist_ok=True)
    cv2.imwrite(IMAGE_PATH, image)

    # Compute rate
    rate = cfg.RESIZE_RATE
    orig_altitude = cfg.PHOTO_ALTITUDE

    cmpx = convert_altitude_to_cmpx(altitude, cfg)
    orig_cmpx = convert_altitude_to_cmpx(orig_altitude, cfg)
    rate = rate * cmpx / orig_cmpx
    print("Resize rate: %.4f" % (rate))

    # Split all images
    split = splitbase(
        IMAGE_DIR, SPLIT_DIR,
        subsize=768, gap=384, ext='.jpg'
    )

    # Resize image before cut
    split.splitdata(rate=rate)

    # Get paths
    image_paths = glob.glob(os.path.join(SPLIT_DIR, '*'))

    return image_paths


def postprocess_results(results):
    RESULT_DIR = os.path.join(TEMP_DIR, 'results')
    MERGE_DIR = os.path.join(TEMP_DIR, 'merge')
    RESTORED_DIR = os.path.join(TEMP_DIR, 'restored')
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(MERGE_DIR, exist_ok=True)
    os.makedirs(RESTORED_DIR, exist_ok=True)

    # Save model results
    for cat in results.keys():
        if cat == 'background':
            continue
        with open(os.path.join(RESULT_DIR, 'Task1_{}.txt'.format(cat)), 'w') as f:
            for img_id in results[cat]:
                for pt in results[cat][img_id]:
                    f.write('{} {:.12f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(
                        img_id, pt[8], pt[0], pt[1], pt[2], pt[3], pt[4], pt[5], pt[6], pt[7]))

    # Merge polygons
    mergebypoly(RESULT_DIR, MERGE_DIR)

    # Format results
    data = Task2groundtruth_poly(MERGE_DIR, RESTORED_DIR)

    return data


def plot_crop_results(orig_image, results, image_id):
    for cat in results.keys():
        if cat == 'background':
            continue

        result = results[cat][image_id]
        for pred in result:
            score = pred[-1]
            tl = np.asarray([pred[0], pred[1]], np.float32)
            tr = np.asarray([pred[2], pred[3]], np.float32)
            br = np.asarray([pred[4], pred[5]], np.float32)
            bl = np.asarray([pred[6], pred[7]], np.float32)

            tt = (np.asarray(tl, np.float32) +
                  np.asarray(tr, np.float32)) / 2
            rr = (np.asarray(tr, np.float32) +
                  np.asarray(br, np.float32)) / 2
            bb = (np.asarray(bl, np.float32) +
                  np.asarray(br, np.float32)) / 2
            ll = (np.asarray(tl, np.float32) +
                  np.asarray(bl, np.float32)) / 2

            box = np.asarray([tl, tr, br, bl], np.float32)
            cen_pts = np.mean(box, axis=0)
            cv2.line(orig_image, (int(cen_pts[0]), int(cen_pts[1])), (int(
                tt[0]), int(tt[1])), (0, 0, 255), 1, 1)
            cv2.line(orig_image, (int(cen_pts[0]), int(cen_pts[1])), (int(
                rr[0]), int(rr[1])), (255, 0, 255), 1, 1)
            cv2.line(orig_image, (int(cen_pts[0]), int(cen_pts[1])), (int(
                bb[0]), int(bb[1])), (0, 255, 0), 1, 1)
            cv2.line(orig_image, (int(cen_pts[0]), int(cen_pts[1])), (int(
                ll[0]), int(ll[1])), (255, 0, 0), 1, 1)

            orig_image = cv2.drawContours(
                orig_image, [np.int0(box)], -1, (255, 0, 255), 1, 1)

            cv2.putText(orig_image, '{:.2f} {}'.format(score, cat), (int(box[1][0]), int(box[1][1])),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 1, 1)

    try:
        cv2.imshow('pr_image', orig_image)
        k = cv2.waitKey(0) & 0xFF
        if k == ord('q'):
            cv2.destroyAllWindows()
            exit(1)
    except:
        results_folder = os.path.join(WORK_DIR, 'results/detections_by_crop')
        os.makedirs(results_folder, exist_ok=True)
        cv2.imwrite(os.path.join(results_folder, image_id), orig_image)


def plot_full_image(objects, img_path, output_path):
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(40, 40))
    plt.imshow(img)
    plt.axis('off')
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for obj in objects:
        c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
        poly = obj['polygon']
        polygons.append(Polygon(poly))
        color.append(c)
    p = PatchCollection(polygons, facecolors=color,
                        linewidths=0, alpha=0.4)
    ax.add_collection(p)
    p = PatchCollection(polygons, facecolors='none',
                        edgecolors=color, linewidths=1)
    ax.add_collection(p)

    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)


def save_results(image_path, results):
    print("Saving results...")
    results_dir = os.path.dirname(image_path)
    image_id = os.path.basename(image_path).rsplit(".", 1)[0]

    count = {}

    print("\nLABELS COUNT:")
    print("--------------------")
    labels = np.array([r['label'] for r in results])
    for label in np.unique(labels):
        count[label] = int(np.sum(labels == label))
        print("%s - %d" % (label, count[label]))
    print("--------------------\n")

    data = {
        "annotations": results,
        "labels_count": count
    }

    output_path = os.path.join(results_dir, image_id + "_detections.jpg")
    plot_full_image(results, image_path, output_path)

    json_path = os.path.join(results_dir, image_id + '.json')
    with open(json_path, 'w') as fp:
        json.dump(data, fp)
