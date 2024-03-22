from omegaconf import OmegaConf
from ..samplers import PLMSSampler
from .io import get_model, export_imgs
import os
import torch
import numpy as np
from torch import autocast
from contextlib import nullcontext
import gc
from ..constants import default_ckpt_path
import glob


def txt2img(
    rank,
    world_size,
    devices,
    prompt,
    output_dir,
    hw,
    ddim_steps,
    scale,
    ddim_eta,
    batch_size,
    variations,
    precision,
    ckpt_path=default_ckpt_path,
):
    verbose = rank == 0
    device = devices if type(devices) == int else devices[rank]

    start_idx = int(variations / world_size * rank)
    variations = (
        variations - start_idx
        if rank + 1 == world_size
        else int(variations / world_size * (rank + 1)) - start_idx
    )

    samples_dir = os.path.join(output_dir, "samples")
    if rank == 0:
        os.makedirs(samples_dir, exist_ok=True)

        with open(os.path.join(output_dir, "prompt.txt"), "w") as handler:
            handler.write(prompt)

    H, W = hw
    C = 4
    f = 8
    start_code = None
    exported_count = 0
    ckpt_exported_count = 0

    model = get_model(None)
    sampler = PLMSSampler(model)
    model.eval()
    model.cuda(device)
    model.cond_stage_model.device = device

    if os.path.isdir(ckpt_path):
        ckpt_paths = sorted(glob.glob(f"{ckpt_path}/*"))
    else:
        ckpt_paths = [ckpt_path]

    c_bs = None
    uc_bs = None
    for ckpt_path in ckpt_paths:

        if len(ckpt_paths) > 1:
            print(f"Loading checkpoint from '{ckpt_path}'")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"], strict=False)
        ckpt_exported_count = 0

        precision_scope = autocast if precision == "autocast" else nullcontext
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    # get text embeddings
                    if c_bs is None:
                        c_bs = model.get_learned_conditioning([prompt] * batch_size)
                        uc_bs = None
                        if scale != 1.0:
                            uc_bs = model.get_learned_conditioning([""] * batch_size)

                    while ckpt_exported_count < variations:
                        local_batch_size = (
                            batch_size
                            if ckpt_exported_count + batch_size <= variations
                            else variations - ckpt_exported_count
                        )
                        c = c_bs[:local_batch_size]
                        uc = uc_bs[:local_batch_size]

                        shape = [C, H // f, W // f]
                        samples_ddim, _ = sampler.sample(
                            S=ddim_steps,
                            conditioning=c,
                            batch_size=local_batch_size,
                            shape=shape,
                            verbose=verbose,
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
                        export_imgs(
                            x_samples_ddim, samples_dir, start_idx + exported_count
                        )
                        ckpt_exported_count += local_batch_size
                        exported_count += local_batch_size

    gc.collect()
    torch.cuda.empty_cache()
