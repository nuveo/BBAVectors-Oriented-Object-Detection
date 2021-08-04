import os
import cv2
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
    plot_crop_results,
    save_results
)


class ObjectDetection:
    def __init__(self, model_dir):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.cfg = load_config(os.path.join(model_dir, 'config.yaml'))
        self.model, self.decoder = load_model(
            model_dir, self.cfg, self.device)

    def predict(self, image_path, altitude, plot=False):
        init_time = time.time()
        categories = self.cfg.CATEGORIES
        results = {cat: defaultdict(list) for cat in categories}

        orig_image = cv2.imread(image_path)
        if orig_image is None:
            return None

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
                plot_crop_results(image_split, results, image_id)

        results = postprocess_results(results)

        print('Total inference time: %.4f' %
              (time.time() - init_time))

        return results


def run_inference(model_dir, image_path, altitude, plot):
    model = ObjectDetection(model_dir)
    preds = model.predict(image_path, altitude=altitude, plot=plot)
    save_results(image_path, preds)
    print("Finished!")
