from torchvision.transforms import (
    Compose,
    CenterCrop,
    ToTensor,
    Lambda,
    functional as F,
    RandomHorizontalFlip,
    ColorJitter,
)
from torch.utils.data import Dataset
import glob
from typing import Tuple
from torch import Tensor
from PIL import Image
from torch.nn import Module
import torch
import random


class RatioSaveRandomResize(Module):
    def __init__(self, size: float, u_variance) -> None:
        super().__init__()
        self.size = size
        self.u_variance = u_variance

    def forward(self, img):
        scale = random.random() * (self.u_variance) + 1
        size = self.size * scale
        if type(img) == torch.Tensor:
            *_, h, w = img.shape
        else:
            w, h = img.size

        if w < h:
            new_w = int(size)
            new_h = int(h / w * size)
        else:
            new_h = int(size)
            new_w = int(w / h * size)
        return F.resize(img, [new_h, new_w])


class DreamBoothDataset(Dataset):
    def __init__(
        self,
        class_img_dir: str,
        instance_img_dir: str,
        hw: Tuple[int, int] = [512, 512],
        class_img_count: int = -1,
        pre_crop_scale: float = 1.1,
        length=None,
    ) -> None:
        assert pre_crop_scale >= 1

        self.class_imgs = sorted(glob.glob(f"{class_img_dir}/*"))
        if class_img_count != -1:
            self.class_imgs = self.class_imgs[:class_img_count]
        self.instance_imgs = sorted(glob.glob(f"{instance_img_dir}/*"))
        self.class_img_cnt = len(self.class_imgs)
        self.instance_img_cnt = len(self.instance_imgs)
        self._len = (
            max(self.class_img_cnt, self.instance_img_cnt) if length is None else length
        )

        self.trans = Compose(
            [
                RandomHorizontalFlip(),
                RatioSaveRandomResize(min(hw), pre_crop_scale - 1),
                CenterCrop(hw),
                ColorJitter(0.3, 0.3, 0.3),
                ToTensor(),
                Lambda(lambda x: x * 2 - 1),
            ]
        )

    def __len__(self):
        return self._len

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        inst_path = self.instance_imgs[index % self.instance_img_cnt]
        clas_path = self.class_imgs[index % self.class_img_cnt]

        inst_img = self.trans(Image.open(inst_path))
        clas_img = self.trans(Image.open(clas_path))

        return inst_img, clas_img
