from torchvision.transforms import (
    Compose,
    CenterCrop,
    ToTensor,
    Lambda,
    functional as F,
)
from torch.utils.data import Dataset
import glob
from typing import Tuple
from torch import Tensor
from PIL import Image
from torch.nn import Module
import torch


class RatioSaveResize(Module):
    def __init__(self, size: float) -> None:
        super().__init__()
        self.size = size

    def forward(self, img):
        if type(img) == torch.Tensor:
            *_, h, w = img.shape
        else:
            h, w = img.size

        if w < h:
            new_w = int(self.size)
            new_h = int(h / w * self.size)
        else:
            new_h = int(self.size)
            new_w = int(w / h * self.size)
        return F.resize(img, [new_h, new_w])


class DreamBoothDataset(Dataset):
    def __init__(
        self,
        class_img_dir: str,
        instance_img_dir: str,
        hw: Tuple[int, int] = [512, 512],
    ) -> None:
        self.class_imgs = glob.glob(f"{class_img_dir}/*")
        self.instance_imgs = glob.glob(f"{instance_img_dir}/*")
        self.class_img_cnt = len(self.class_imgs)
        self.instance_img_cnt = len(self.instance_imgs)
        self._len = max(self.class_img_cnt, self.instance_img_cnt)
        self.trans = Compose(
            [
                RatioSaveResize(min(hw)),
                CenterCrop(hw),
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
