from omegaconf import OmegaConf
from ..samplers import PLMSSampler
from .io import get_model, export_imgs
import os
import torch
import numpy as np
from torch import autocast
from contextlib import nullcontext


def txt2img(
    rank,
    world_size,
    prompt,
    output_dir,
    hw,
    ddim_steps,
    scale,
    ddim_eta,
    batch_size,
    variations,
    precision,
):
    verbose = rank == 0

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

    model = get_model()
    model.cuda(rank)
    model.cond_stage_model.device = rank
    sampler = PLMSSampler(model)

    H, W = hw
    C = 4
    f = 8
    start_code = None
    model = sampler.model
    exported_count = 0

    precision_scope = autocast if precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                while exported_count < variations:
                    local_batch_size = (
                        batch_size
                        if exported_count + batch_size <= variations
                        else variations - exported_count
                    )

                    c = model.get_learned_conditioning([prompt] * local_batch_size)
                    uc = None
                    if scale != 1.0:
                        uc = model.get_learned_conditioning(local_batch_size * [""])

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
                    export_imgs(x_samples_ddim, samples_dir, start_idx + exported_count)
                    exported_count += local_batch_size
