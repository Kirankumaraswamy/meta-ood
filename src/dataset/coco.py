import json
import os
import random

import torch
from PIL import Image
from typing import Callable, Optional
from torch.utils.data import Dataset
import numpy as np


class COCO(Dataset):

    train_id_in = 255
    train_id_out = 254
    min_image_size = 480

    def __init__(self, root: str, split: str = "train", transform: Optional[Callable] = None, shuffle=True,
                 proxy_size: Optional[int] = None) -> None:
        """
        COCO dataset loader
        """
        self.root = root
        self.coco_year = list(filter(None, self.root.split("/")))[-1]
        self.split = split + self.coco_year
        self.images = []
        self.targets = []
        self.transform = transform
        self.coco_data_dicts = []

        gt_json_file = os.path.join(self.root, "annotations", "panoptic_" + self.split+".json")
        json_data = json.load(open(gt_json_file))

        for key, value in json_data.items():
            data = {}
            data["file_name"] = value["file_name"]
            data["image_id"] = key.split(".")[0]
            data["pan_seg_file_name"] = value["pan_seg_file_name"]
            data["sem_seg_file_name"] = value["sem_seg_file_name"]
            data["segments_info"] = value["segments_info"]
            data["dataset"] = "coco"
            self.coco_data_dicts.append(data)

        '''for root, _, filenames in os.walk(os.path.join(self.root, "annotations", "ood_seg_" + self.split)):
            assert self.split in ['train' + self.coco_year, 'val' + self.coco_year]
            for filename in filenames:
                if os.path.splitext(filename)[-1] == '.png':
                    #self.targets.append(os.path.join(root, filename))
                    #self.images.append(os.path.join(self.root, self.split, filename.split(".")[0] + ".jpg"))
                    data = {}
                    data["file_name"] = os.path.join(self.root, self.split, filename.split(".")[0] + ".jpg")
                    data["image_id"] = filename.split(".")[0]
                    data["pan_seg_file_name"] = os.path.join(root, filename)
                    data["dataset"] = "coco"
                    self.coco_data_dicts.append(data)'''


        """
        shuffle data and subsample
        """
        if shuffle:
            random.shuffle(self.coco_data_dicts)
        if proxy_size is not None:
            self.coco_data_dicts = list(self.coco_data_dicts[:int(proxy_size)])

    def __len__(self):
        """Return total number of images in the whole dataset."""
        return len(self.coco_data_dicts)

    def __getitem__(self, i):
        """Return raw image and ground truth in PIL format or as torch tensor"""
        image = Image.open(self.coco_data_dicts[i]["file_name"]).convert('RGB')
        target = Image.open(self.coco_data_dicts[i]["seg_file_name"]).convert('L')
        if self.transform is not None:
            image, target = self.transform(image, target)
        return image, target

    def __repr__(self):
        """Return number of images in each dataset."""
        fmt_str = 'Number of COCO Images: %d\n' % len(self.images)
        return fmt_str.strip()

    def panoptic_target_generator(self, pan_seg_gt):
        targets = {}
        height, width = pan_seg_gt.shape[0], pan_seg_gt.shape[1]
        semantic = np.zeros_like(pan_seg_gt, dtype=np.uint8) + self.ignore_label
        center = np.zeros((height, width), dtype=np.float32)
        center_pts = []
        offset = np.zeros((2, height, width), dtype=np.float32)
        y_coord, x_coord = np.meshgrid(
            np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij"
        )
        # Generate pixel-wise loss weights
        semantic_weights = np.ones_like(pan_seg_gt, dtype=np.uint8)
        # 0: ignore, 1: has instance
        center_weights = np.zeros_like(pan_seg_gt, dtype=np.uint8)
        offset_weights = np.zeros_like(pan_seg_gt, dtype=np.uint8)

        unique = torch.unique(pan_seg_gt).numpy()
        unique = [i for i in unique]
        for i in unique:
            if i > 0:
                mask_index = np.where(pan_seg_gt == i)

                # Find instance area
                ins_area = len(mask_index[0])
                if ins_area < self.small_instance_area:
                    semantic_weights[panoptic == seg["id"]] = self.small_instance_weight

                center_y, center_x = np.mean(mask_index[0]), np.mean(mask_index[1])
                center_pts.append([center_y, center_x])

                # generate center heatmap
                y, x = int(round(center_y)), int(round(center_x))
                sigma = self.sigma
                # upper left
                ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
                # bottom right
                br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

                # start and end indices in default Gaussian image
                gaussian_x0, gaussian_x1 = max(0, -ul[0]), min(br[0], width) - ul[0]
                gaussian_y0, gaussian_y1 = max(0, -ul[1]), min(br[1], height) - ul[1]

                # start and end indices in center heatmap image
                center_x0, center_x1 = max(0, ul[0]), min(br[0], width)
                center_y0, center_y1 = max(0, ul[1]), min(br[1], height)
                center[center_y0:center_y1, center_x0:center_x1] = np.maximum(
                    center[center_y0:center_y1, center_x0:center_x1],
                    self.g[gaussian_y0:gaussian_y1, gaussian_x0:gaussian_x1],
                )

                # generate offset (2, h, w) -> (y-dir, x-dir)
                offset[0][mask_index] = center_y - y_coord[mask_index]
                offset[1][mask_index] = center_x - x_coord[mask_index]


