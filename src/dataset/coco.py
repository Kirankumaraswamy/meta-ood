import os
import random

from PIL import Image
from typing import Callable, Optional
from torch.utils.data import Dataset


class COCO(Dataset):

    train_id_in = 0
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

        for root, _, filenames in os.walk(os.path.join(self.root, "annotations", "ood_seg_" + self.split)):
            assert self.split in ['train' + self.coco_year, 'val' + self.coco_year]
            for filename in filenames:
                if os.path.splitext(filename)[-1] == '.png':
                    #self.targets.append(os.path.join(root, filename))
                    #self.images.append(os.path.join(self.root, self.split, filename.split(".")[0] + ".jpg"))
                    data = {}
                    data["file_name"] = os.path.join(self.root, self.split, filename.split(".")[0] + ".jpg")
                    data["image_id"] = filename.split(".")[0]
                    data["sem_seg_file_name"] = os.path.join(root, filename)
                    data["dataset"] = "coco"
                    self.coco_data_dicts.append(data)


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
