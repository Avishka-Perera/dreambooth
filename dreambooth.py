from argparse import ArgumentParser
import ast
import os
from src.util.diffusion import txt2img
from src.util.io import get_output_path, save_args, get_model
import torch
from torch import multiprocessing as mp
from src.dataset import DreamBoothDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import gc
import yaml


def get_latest_ckpt(ckpt_dir):
    ckpt_lst = sorted(os.listdir(ckpt_dir))
    if len(ckpt_lst) > 0:
        return os.path.join(ckpt_dir, ckpt_lst[-1])
    else:
        return None


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--class-name",
        type=str,
        required=True,
        help="Class name of the given instance",
    )
    parser.add_argument(
        "-i",
        "--instance-img-dir",
        type=str,
        required=True,
        help="Directory to save outputs",
    )
    parser.add_argument(
        "-d",
        "--devices",
        type=ast.literal_eval,
        nargs="+",
        help="CUDA Devices that must be used for training",
        default=list(range(torch.cuda.device_count())),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="out/dreambooth",
        help="Directory to save outputs",
    )
    parser.add_argument(
        "-r",
        "--resume-dir",
        default=None,
        help="Directory to start the job",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast",
    )

    # class image generation parameters
    parser.add_argument(
        "--generation-batch-size",
        "--gb",
        type=int,
        default=4,
        help="Class image generation batch size",
    )
    parser.add_argument(
        "--hw",
        type=ast.literal_eval,
        nargs="+",
        help="Height and the Width of the required image",
        default=[512, 512],
    )
    parser.add_argument(
        "--ddim-steps",
        type=int,
        default=50,
        help="",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="",
    )
    parser.add_argument(
        "--ddim-eta",
        type=float,
        default=0,
        help="",
    )
    parser.add_argument(
        "--class-img-count",
        type=int,
        default=1000,
        help="Number class images to use for training",
    )

    # train parameters
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "-l",
        "--lamb",
        type=float,
        default=1.0,
        help="",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Number of iterations to be trained.",
    )
    parser.add_argument(
        "-s",
        "--save-every",
        type=int,
        default=-1,
        help="Save checkpoints after each this much of iterations",
    )
    parser.add_argument(
        "-t",
        "--train-text-encoder",
        action="store_true",
        default=False,
        help="Whether to train the text encoder",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    assert os.path.exists(args.instance_img_dir)

    if args.resume_dir is None:
        output_dir = get_output_path(args.output_dir, "run")
        os.makedirs(output_dir)
    else:
        output_dir = args.resume_dir
    save_args(output_dir, args)

    class_prompt = f"A {args.class_name}"
    instance_prompt = f"A sks {args.class_name}"

    # generate from class prompt
    class_img_dir = os.path.join(output_dir, "class-imgs")
    if os.path.exists(class_img_dir):
        if (
            len(os.listdir(os.path.join(class_img_dir, "samples")))
            != args.class_img_count
        ):
            raise NotImplementedError(
                "Resuming of image generation not implemented yet"
            )
    else:
        print(
            f"Exporting class images for the prompt '{class_prompt}', to directory {class_img_dir}"
        )
        world_size = len(args.devices)
        mp.spawn(
            txt2img,
            (
                world_size,
                args.devices,
                class_prompt,
                class_img_dir,
                args.hw,
                args.ddim_steps,
                args.scale,
                args.ddim_eta,
                args.generation_batch_size,
                args.class_img_count,
                args.precision,
            ),
            nprocs=world_size,
            join=True,
        )
        gc.collect()
        torch.cuda.empty_cache()
        print("Class image exporting done!")

    device = args.devices[0]
    ckpt_dir = os.path.join(output_dir, "ckpts")
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # setup model
    model = get_model()
    model.train()
    model = model.to(device)
    model.cond_stage_model.device = device
    for nm, param in model.named_parameters():
        assert param.device.index == device, (nm, param.device)
    optim_params = (
        (*model.model.parameters(), *model.cond_stage_model.parameters())
        if args.train_text_encoder
        else model.model.parameters()
    )
    optimizer = Adam(optim_params, lr=args.learning_rate)
    latest_ckpt_path = get_latest_ckpt(ckpt_dir)
    if args.resume_dir is not None and latest_ckpt_path is not None:
        print(f"Loading checkpoint from {latest_ckpt_path}")
        ckpt = torch.load(latest_ckpt_path)
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_it = ckpt["iteration"] + 1
    else:
        start_it = 0

    freeze = (
        [model.first_stage_model]
        if args.train_text_encoder
        else [model.cond_stage_model, model.first_stage_model]
    )
    for f_m in freeze:
        for param in f_m.parameters():
            param.requires_grad = False

    # dataset
    ds = DreamBoothDataset(
        os.path.join(class_img_dir, "samples"),
        args.instance_img_dir,
        args.hw,
        class_img_count=args.class_img_count,
        length=args.iterations - start_it,
    )
    dl = DataLoader(ds, 1, shuffle=True)
    with torch.no_grad():
        clas_prompt_enc = model.get_learned_conditioning(class_prompt)
        inst_prompt_enc = model.get_learned_conditioning(instance_prompt)

    tb_writer = SummaryWriter(log_dir)
    print("Starting Dreambooth training...")
    data_len = len(dl)
    with tqdm(
        total=data_len + start_it,
        postfix={"loss": "undefined"},
    ) as pbar:
        pbar.update(start_it)
        for i, (inst_img, clas_img) in enumerate(dl):
            optimizer.zero_grad()

            inst_img, clas_img = inst_img.to(device), clas_img.to(device)
            # move to latent space
            with torch.no_grad():
                latent_inst_img = model.get_first_stage_encoding(
                    model.encode_first_stage(inst_img)
                )
                latent_clas_img = model.get_first_stage_encoding(
                    model.encode_first_stage(clas_img)
                )

            inst_loss, _ = model(latent_inst_img, inst_prompt_enc)
            clas_loss, _ = model(latent_clas_img, clas_prompt_enc)
            loss = inst_loss + args.lamb * clas_loss

            loss.backward()
            optimizer.step()

            loss = loss.cpu().item()
            pbar.set_postfix({"loss": loss})
            tb_writer.add_scalar("Loss", loss, i + start_it)

            if (
                args.save_every != -1
                and (i + start_it) % args.save_every == 0
                and i + start_it != data_len - 1
                and (i != 0)
            ):
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "iteration": i + start_it,
                    },
                    os.path.join(ckpt_dir, f"i{i+start_it:>05}.ckpt"),
                )

            pbar.update()

        torch.save(
            {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iteration": i + start_it,
            },
            os.path.join(ckpt_dir, f"i{i+start_it:>05}.ckpt"),
        )

    print("Training complete!")
