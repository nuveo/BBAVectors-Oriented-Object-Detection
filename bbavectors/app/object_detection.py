import os
import sys
import cv2
import json
import time
import torch
import argparse
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from bbavectors.configs import load_config
from bbavectors.datasets.base import BaseDataset
from bbavectors.func_utils import non_maximum_suppression
from bbavectors.app.utils import (
    decode_prediction,
    load_model,
    generate_splits,
    postprocess_results,
    clear_temp_folder,
    plot_results
)


class ObjectDetection:
    def __init__(self, model_dir):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.cfg = load_config(os.path.join(model_dir, 'config.yaml'))
        self.model, self.decoder = load_model(
            model_dir, self.cfg, self.device)

    def predict(self, orig_image, altitude, plot=False):
        init_time = time.time()
        categories = self.cfg.CATEGORIES
        results = {cat: defaultdict(list) for cat in categories}

        if orig_image is None:
            return results

        print("Generating image splits. This may take a while.")
        image_paths = generate_splits(orig_image, altitude, self.cfg)
        del orig_image

        print("Start inference.")
        for path in tqdm(image_paths):
            image_split = cv2.imread(path)
            if image_split is None:
                continue

            # Preprocess input image
            input_h, input_w, _ = self.cfg.INPUT_SHAPE
            image = BaseDataset.processing_test(image_split, input_h, input_w)
            image = image.to(self.device)

            # Do inference
            with torch.no_grad():
                pr_decs = self.model(image)

            # Decode
            decoded_pts = []
            decoded_scores = []
            predictions = self.decoder.ctdet_decode(pr_decs)
            pts0, scores0 = decode_prediction(
                image_split.shape, predictions, self.cfg)
            decoded_pts.append(pts0)
            decoded_scores.append(scores0)

            # Apply NMS and generate final results
            image_id = os.path.basename(path)
            for cat in categories:
                if cat == 'background':
                    continue
                pts_cat = []
                scores_cat = []
                for pts0, scores0 in zip(decoded_pts, decoded_scores):
                    pts_cat.extend(pts0[cat])
                    scores_cat.extend(scores0[cat])
                pts_cat = np.asarray(pts_cat, np.float32)
                scores_cat = np.asarray(scores_cat, np.float32)
                if pts_cat.shape[0]:
                    nms_results = non_maximum_suppression(pts_cat, scores_cat)
                    results[cat][image_id].extend(nms_results)

            if plot:
                plot_results(image_split, results, image_id)

        results = postprocess_results(results)

        print('Total inference time: %.4f' %
              (time.time() - init_time))

        clear_temp_folder()

        return results


def parse_args():
    parser = argparse.ArgumentParser(description='BBAVectors')
    parser.add_argument('--model_dir', required=True,
                        type=str, help='Model weights directory.')
    parser.add_argument('--image', required=True,
                        type=str, help='Full image path.')
    parser.add_argument('--altitude', default=120,
                        type=int, help='Drone photo altitude in meters.')
    parser.add_argument('--plot', default=False, action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model = ObjectDetection(args.model_dir)
    img = cv2.imread(args.image)

    anns = model.predict(
        img, altitude=args.altitude, plot=args.plot)

    print("Saving results...")
    result_path = args.image.rsplit(".", 1)[0] + ".json"
    with open(result_path, 'w') as fp:
        json.dump({"annotations": anns}, fp)

    print("Finished!")
