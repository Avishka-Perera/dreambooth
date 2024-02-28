import numpy as np
import torch
from PIL import Image
from .util import instantiate_from_config
import os
import sys
from io import StringIO
from contextlib import nullcontext
from omegaconf import OmegaConf


def get_output_path(root: str, lead: str):
    if os.path.exists(root):
        i = 0
        path = os.path.join(root, f"{lead}{i}")
        while os.path.exists(path):
            i += 1
            path = os.path.join(root, f"{lead}{i}")
    else:
        path = os.path.join(root, f"{lead}0")

    return path


def get_model(verbose=False):
    config_path = "configs/stable-diffusion-v1.yaml"
    ckpt = "weights/model.ckpt"

    config = OmegaConf.load(config_path)
    if verbose:
        print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd and verbose:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]

    context = nullcontext if verbose else PrintSuppressContext
    with context():
        model = instantiate_from_config(config.model)

    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


def export_imgs(imgs: np.ndarray, dir: str, start: int = 0) -> None:
    for i in range(len(imgs)):
        im = Image.fromarray(imgs[i])
        im.save(os.path.join(dir, f"{i+start:04}.jpg"))


class PrintSuppressContext:
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = StringIO()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self._stdout
        if exc_type is not None:
            return False
