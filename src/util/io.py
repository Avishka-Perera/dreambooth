import numpy as np
import torch
from PIL import Image
from .util import instantiate_from_config
import os
import sys
from io import StringIO
from contextlib import nullcontext
from omegaconf import OmegaConf
import glob
import yaml
from ..constants import default_ckpt_path, config_path


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


def get_args_path(dir):
    args_paths = glob.glob(f"{dir}/args**.yaml")
    if len(args_paths) == 0:
        return f"{dir}/args.yaml"
    i = 2
    while f"{dir}/args{i}.yaml" in args_paths:
        i += 1
    args_path = f"{dir}/args{i}.yaml"
    return args_path


def save_args(output_dir, args):
    args_save_path = get_args_path(output_dir)
    with open(args_save_path, "w") as handler:
        yaml.dump(vars(args), handler)


def get_model(ckpt=default_ckpt_path, verbose=False):

    config = OmegaConf.load(config_path)
    context = nullcontext if verbose else PrintSuppressContext
    with context():
        model = instantiate_from_config(config.model)

    if ckpt is not None:
        if verbose:
            print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        if "global_step" in pl_sd and verbose:
            print(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]

        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0 and verbose:
            print("missing keys:")
            print(m)
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u)

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
