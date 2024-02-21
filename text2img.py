from argparse import ArgumentParser
import torch
from src.util.io import get_output_path, load_model_from_config, export_imgs
from omegaconf import OmegaConf
from src.samplers import PLMSSampler
import os
import numpy as np
from torch import autocast
from contextlib import nullcontext


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "-p", "--prompt", type=str, required=True, help="Prompt to convert to image"
    )
    parser.add_argument(
        "-v",
        "--variations",
        type=int,
        default=1,
        help="Number of variations to be generated",
    )
    parser.add_argument(
        "-o", "--output-dir", default="out", help="Directory to save outputs"
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast",
    )

    return parser.parse_args()


def main(args):

    output_dir = get_output_path(args.output_dir, "inf")
    samples_dir = os.path.join(output_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    with open(os.path.join(output_dir, "prompt.txt"), "w") as handler:
        handler.write(args.prompt)

    config_path = "configs/v1-inference.yaml"
    model_path = "weights/model.ckpt"

    config = OmegaConf.load(config_path)
    model = load_model_from_config(config, model_path)
    sampler = PLMSSampler(model)

    data = [args.prompt] * args.variations

    C = 4
    H = 512
    W = 512
    f = 8
    ddim_steps = 50
    scale = 7.5
    ddim_eta = 0
    start_code = None
    batch_size = len(data)

    precision_scope = autocast if args.precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                c = model.get_learned_conditioning(data)
                uc = None
                if scale != 1.0:
                    uc = model.get_learned_conditioning(batch_size * [""])

                shape = [C, H // f, W // f]
                samples_ddim, _ = sampler.sample(
                    S=ddim_steps,
                    conditioning=c,
                    batch_size=batch_size,
                    shape=shape,
                    verbose=False,
                    unconditional_guidance_scale=scale,
                    unconditional_conditioning=uc,
                    eta=ddim_eta,
                    x_T=start_code,
                )

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp(
                    (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0
                )
                x_samples_ddim = (
                    x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy() * 255
                ).astype(np.uint8)

    export_imgs(x_samples_ddim, samples_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
