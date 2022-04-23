import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from src.dataset.coco import COCO
from src.dataset.cityscapes import Cityscapes


class CityscapesCocoMix(Dataset):

    def __init__(self, split='train', transform=None,
                 cs_root="/home/datasets/cityscapes",
                 coco_root="/home/datasets/COCO/2017",
                 subsampling_factor=0.1, cs_split=None, coco_split=None,):

        self.transform = transform
        if cs_split is None or coco_split is None:
            self.cs_split = split
            self.coco_split = split
        else:
            self.cs_split = cs_split
            self.coco_split = coco_split

        self.cs = Cityscapes(root=cs_root, split=self.cs_split)
        self.coco = COCO(root=coco_root, split=self.coco_split, proxy_size=int(subsampling_factor*len(self.cs)))
        self.data_dicts = self.cs.cityscapes_data_dicts + self.coco.coco_data_dicts
        self.train_id_out = self.coco.train_id_out
        self.num_classes = self.cs.num_train_ids
        self.mean = self.cs.mean
        self.std = self.cs.std
        self.void_ind = self.cs.ignore_in_eval_ids

    def __getitem__(self, i):
        data = self.data_dicts[i]
        image = Image.open(data["file_name"]).convert('RGB')
        sem_seg = Image.open(data["sem_seg_file_name"]).convert('L')

        if self.transform is not None:
            image, sem_seg = self.transform(image, sem_seg)
        data["image"] = image
        data["sem_seg"] = sem_seg
        return data, sem_seg

    def __len__(self):
        """Return total number of images in the whole dataset."""
        return len(self.data_dicts)

    def __repr__(self):
        """Return number of images in each dataset."""
        fmt_str = 'Cityscapes Split: %s\n' % self.cs_split
        fmt_str += '----Number of images: %d\n' % len(self.cs)
        fmt_str += 'COCO Split: %s\n' % self.coco_split
        fmt_str += '----Number of images: %d\n' % len(self.coco)
        return fmt_str.strip()

