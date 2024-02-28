from torchvision.transforms import Compose, CenterCrop, ToTensor, Lambda
from torch.utils.data import Dataset
import glob
from typing import Tuple
from torch import Tensor
from PIL import Image
import random


class DreamBoothDataset(Dataset):
    def __init__(
        self, class_img_dir: str, instance_img_dir: str, hw: Tuple[int, int]
    ) -> None:
        self.class_imgs = glob.glob(f"{class_img_dir}/*")
        self.instance_imgs = glob.glob(f"{instance_img_dir}/*")
        self.len = len(self.instance_imgs)
        self.trans = Compose([CenterCrop(hw), ToTensor(), Lambda(lambda x: x * 2 - 1)])

    def __len__(self):
        return self.len

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        inst_path = self.instance_imgs[index]
        clas_path = random.choice(self.class_imgs)

        inst_img = self.trans(Image.open(inst_path))
        clas_img = self.trans(Image.open(clas_path))

        return inst_img, clas_img
