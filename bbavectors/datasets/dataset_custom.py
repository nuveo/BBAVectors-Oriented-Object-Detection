from .base import BaseDataset
import os
import cv2
import glob
import numpy as np
from DOTA_devkit.ResultMerge_multi_process import mergebypoly


class CUSTOM(BaseDataset):
    def __init__(self, data_dir, phase, input_h=None, input_w=None, down_ratio=None):
        super(CUSTOM, self).__init__(
            data_dir, phase, input_h, input_w, down_ratio)
        self.category = [
            'small-vehicle',
            'medium-vehicle',
            'large-vehicle'
        ]
        self.color_pans = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
        ]
        self.num_classes = len(self.category)
        self.cat_ids = {cat: i for i, cat in enumerate(self.category)}
        self.img_ids = self.load_img_ids()
        self.image_path = os.path.join(data_dir, 'images')
        self.label_path = os.path.join(data_dir, 'labelTxt')

    def load_img_ids(self):
        files = os.listdir(os.path.join(self.data_dir, 'images'))
        image_lists = [f.strip().rsplit(".", 1)[0] for f in files]
        return image_lists

    def load_image(self, index):
        img_id = self.img_ids[index]
        imgFile = glob.glob(os.path.join(self.image_path, img_id+'.*'))[0]
        assert os.path.exists(imgFile), 'image {} not existed'.format(imgFile)
        img = cv2.imread(imgFile)
        return img

    def load_annoFolder(self, img_id):
        return os.path.join(self.label_path, img_id+'.txt')

    def load_annotation(self, index):
        image = self.load_image(index)
        h, w, c = image.shape
        valid_pts = []
        valid_cat = []
        valid_dif = []
        with open(self.load_annoFolder(self.img_ids[index]), 'r') as f:
            for i, line in enumerate(f.readlines()):
                obj = line.split(' ')  # list object
                if len(obj) > 8:
                    x1 = min(max(float(obj[0]), 0), w - 1)
                    y1 = min(max(float(obj[1]), 0), h - 1)
                    x2 = min(max(float(obj[2]), 0), w - 1)
                    y2 = min(max(float(obj[3]), 0), h - 1)
                    x3 = min(max(float(obj[4]), 0), w - 1)
                    y3 = min(max(float(obj[5]), 0), h - 1)
                    x4 = min(max(float(obj[6]), 0), w - 1)
                    y4 = min(max(float(obj[7]), 0), h - 1)
                    # TODO: filter small instances
                    xmin = max(min(x1, x2, x3, x4), 0)
                    xmax = max(x1, x2, x3, x4)
                    ymin = max(min(y1, y2, y3, y4), 0)
                    ymax = max(y1, y2, y3, y4)
                    if ((xmax - xmin) > 10) and ((ymax - ymin) > 10):
                        valid_pts.append(
                            [[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
                        valid_cat.append(self.cat_ids[obj[8]])
                        valid_dif.append(int(obj[9]))
        f.close()
        annotation = {}
        annotation['pts'] = np.asarray(valid_pts, np.float32)
        annotation['cat'] = np.asarray(valid_cat, np.int32)
        annotation['dif'] = np.asarray(valid_dif, np.int32)
        return annotation

    def merge_crop_image_results(self, result_path, merge_path):
        mergebypoly(result_path, merge_path)
