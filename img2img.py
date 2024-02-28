import numpy as np
from src.util.io import get_output_path, load_img, get_model, export_imgs
from omegaconf import OmegaConf
from src.samplers import DDIMSampler
import os, shutil
import torch
from einops import repeat
from torch import autocast
from contextlib import nullcontext
from tqdm import tqdm, trange
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "-d", "--device", type=int, default=0, help="The device to run the job"
    )
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
        "-i",
        "--init-img",
        type=str,
        required=True,
        help="Initial image to convert",
    )
    parser.add_argument(
        "-o", "--output-dir", default="out/img2img", help="Directory to save outputs"
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast",
    )
    parser.add_argument(
        "-s",
        "--strength",
        type=float,
        help="Influence of the text prompt",
        default=0.75,
    )

    return parser.parse_args()


def main(args):
    model = get_model()
    model.eval()
    model.to(args.device)
    sampler = DDIMSampler(model)

    output_dir = get_output_path(args.output_dir, "run")
    samples_dir = os.path.join(output_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    with open(os.path.join(output_dir, "prompt.txt"), "w") as handler:
        handler.write(args.prompt)
    shutil.copy(
        args.init_img, os.path.join(output_dir, os.path.split(args.init_img)[1])
    )

    # opt
    batch_size = 1
    ddim_steps = 50
    ddim_eta = 0.0
    data = [args.prompt] * batch_size
    scale = 5.0

    assert os.path.isfile(args.init_img)
    init_image = load_img(args.init_img).to(args.device)
    init_image = repeat(init_image, "1 ... -> b ...", b=batch_size)
    init_latent = model.get_first_stage_encoding(
        model.encode_first_stage(init_image)
    )  # move to latent space

    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)

    assert 0.0 <= args.strength <= 1.0, "can only work with strength in [0.0, 1.0]"
    t_enc = int(args.strength * ddim_steps)
    print(f"target t_enc is {t_enc} steps")

    precision_scope = autocast if args.precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                all_samples = list()
                for n in trange(args.variations, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)

                        # encode (scaled latent)
                        z_enc = sampler.stochastic_encode(
                            init_latent,
                            torch.tensor([t_enc] * batch_size).to(args.device),
                        )
                        # decode it
                        samples = sampler.decode(
                            z_enc,
                            c,
                            t_enc,
                            unconditional_guidance_scale=scale,
                            unconditional_conditioning=uc,
                        )

                        x_samples = model.decode_first_stage(samples)
                        x_samples = torch.clamp(
                            (x_samples + 1.0) / 2.0, min=0.0, max=1.0
                        )
                        all_samples.append(x_samples)

    all_samples = (
        torch.concat(all_samples).cpu().permute(0, 2, 3, 1).numpy() * 255
    ).astype(np.uint8)
    export_imgs(all_samples, samples_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
