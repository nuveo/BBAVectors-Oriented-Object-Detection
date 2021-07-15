import os
import cv2
import time
import torch
import numpy as np
from bbavectors.configs import load_config
from bbavectors.datasets.base import BaseDataset
from bbavectors.func_utils import non_maximum_suppression
from bbavectors.app.utils import (
    decode_prediction, load_model
)


class ObjectDetection:
    def __init__(self, model_dir):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.cfg = load_config(os.path.join(model_dir, 'config.yaml'))
        self.model, self.decoder = load_model(
            model_dir, self.cfg, self.device)

    def predict(self, image):
        init_time = time.time()
        categories = self.cfg.CATEGORIES
        results = {cat: [] for cat in categories}

        if image is None:
            return results

        # Preprocess input image
        orig_shape = image.shape
        input_h, input_w, _ = self.cfg.INPUT_SHAPE
        image = BaseDataset.processing_test(image, input_h, input_w)
        image = image.to(self.device)

        # Do inference
        with torch.no_grad():
            pr_decs = self.model(image)

        # Decode
        decoded_pts = []
        decoded_scores = []
        torch.cuda.synchronize()
        predictions = self.decoder.ctdet_decode(pr_decs)
        pts0, scores0 = decode_prediction(orig_shape, predictions, self.cfg)
        decoded_pts.append(pts0)
        decoded_scores.append(scores0)

        # Apply NMS and generate final results
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
                results[cat].extend(nms_results)

            print('Inference time elapsed: %.4f' % (time.time() - init_time))

        return results


if __name__ == "__main__":
    model_path = '/home/guilherme/Documents/Code/Nuveo/BBAVectors-Oriented-Object-Detection/bbavectors/work_dir/weights'
    model = ObjectDetection(model_path)

    img_path = '/home/guilherme/Documents/Code/Nuveo/datasets/der/15m.jpg'
    img = cv2.imread(img_path)

    anns = model.predict(img)
    print(anns)
