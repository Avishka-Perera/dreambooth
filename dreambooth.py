from argparse import ArgumentParser
import ast
import os
from src.util.diffusion import txt2img
from src.util.io import get_output_path, get_model
import torch
from torch import multiprocessing as mp
from src.dataset import DreamBoothDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import Adam


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
        "-b",
        "--batch-size",
        type=int,
        default=4,
        help="Generation batch size",
    )
    parser.add_argument(
        "-v",
        "--variations",
        type=int,
        default=1000,
        help="Number of variations to be generated from the class prompt",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="out/dreambooth",
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast",
    )

    # generation parameters
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

    # train parameters
    parser.add_argument(
        "-l",
        "--lamb",
        type=float,
        default=1.0,
        help="",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=1000,
        help="",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    assert os.path.exists(args.instance_img_dir)

    output_dir = get_output_path(args.output_dir, "run")
    os.makedirs(output_dir)

    # generate from class prompt
    class_img_dir = os.path.join(output_dir, "class-imgs")
    class_prompt = f"A {args.class_name}"
    instance_prompt = f"An sks {args.class_name}"
    print(
        f"Exporting class images for the prompt '{class_prompt}', to directory {class_img_dir}"
    )
    world_size = len(args.devices)
    mp.spawn(
        txt2img,
        (
            world_size,
            class_prompt,
            class_img_dir,
            args.hw,
            args.ddim_steps,
            args.scale,
            args.ddim_eta,
            args.batch_size,
            args.variations,
            args.precision,
        ),
        nprocs=world_size,
        join=True,
    )
    print("Class image exporting done!")

    device = args.devices[0]
    ds = DreamBoothDataset(
        os.path.join(class_img_dir, "samples"), args.instance_img_dir, args.hw
    )
    dl = DataLoader(ds, 1, shuffle=True)

    model = get_model()
    model.to(device)
    model.train()

    freeze = [model.cond_stage_model, model.first_stage_model]
    for f_m in freeze:
        for param in f_m.parameters():
            param.requires_grad = False
    optimizer = Adam(model.model.parameters())

    with torch.no_grad():
        clas_prompt_enc = model.get_learned_conditioning(class_prompt).to(device)
        inst_prompt_enc = model.get_learned_conditioning(instance_prompt).to(device)

    with tqdm(
        total=args.epochs, postfix={"loss": "undefined"}, desc="Training"
    ) as pbar:
        for epoch in range(args.epochs):
            for inst_img, clas_img in dl:

                # move to latent space
                with torch.no_grad():
                    latent_inst_img = model.get_first_stage_encoding(
                        model.encode_first_stage(inst_img.to(device))
                    )
                    latent_clas_img = model.get_first_stage_encoding(
                        model.encode_first_stage(clas_img.to(device))
                    )

                inst_loss, _ = model(latent_inst_img, inst_prompt_enc)
                clas_loss, _ = model(latent_clas_img, clas_prompt_enc)
                loss = inst_loss + args.lamb * clas_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix({"loss": loss.cpu().item()})

            pbar.update()

    ckpt_path = os.path.join(output_dir, "model.ckpt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Training comlpete! Weights saved to '{ckpt_path}'")
