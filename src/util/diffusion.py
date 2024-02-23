import torch
import numpy as np
from torch import autocast
from contextlib import nullcontext
from .io import export_imgs


def sample_to_dir(
    sampler,
    prompt,
    samples_dir,
    HW,
    ddim_steps,
    scale,
    ddim_eta,
    batch_size,
    count,
    precision,
):
    H, W = HW
    C = 4
    f = 8
    start_code = None
    model = sampler.model
    exported_count = 0

    precision_scope = autocast if precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                while exported_count < count:
                    local_batch_size = (
                        batch_size
                        if exported_count + batch_size <= count
                        else count - exported_count
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
                    exported_count += local_batch_size

    export_imgs(x_samples_ddim, samples_dir)
