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
from torch.utils.tensorboard import SummaryWriter
import gc


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

    # generation parameters
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
        "-e",
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs to be trained.",
    )
    parser.add_argument(
        "--class-img-count",
        type=int,
        default=-1,
        help="Number class images to use for training",
    )
    parser.add_argument(
        "-s",
        "--save-every",
        type=int,
        default=-1,
        help="Save checkpoints after each this much of iterations",
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

    class_prompt = f"A {args.class_name}"
    instance_prompt = f"A sks {args.class_name}"

    # generate from class prompt
    class_img_dir = os.path.join(output_dir, "class-imgs")
    if os.path.exists(class_img_dir):
        if len(os.listdir(os.path.join(class_img_dir, "samples"))) != args.variations:
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
                args.variations,
                args.precision,
            ),
            nprocs=world_size,
            join=True,
        )
        gc.collect()
        torch.cuda.empty_cache()
        print("Class image exporting done!")

    device = args.devices[0]

    # setup model
    model = get_model()
    model.train()
    model = model.to(device)
    model.cond_stage_model.device = device
    for nm, param in model.named_parameters():
        assert param.device.index == device, (nm, param.device)
    optimizer = Adam(model.model.parameters(), lr=args.learning_rate)

    freeze = [model.cond_stage_model, model.first_stage_model]
    for f_m in freeze:
        for param in f_m.parameters():
            param.requires_grad = False

    # dataset
    ds = DreamBoothDataset(
        os.path.join(class_img_dir, "samples"),
        args.instance_img_dir,
        args.hw,
        class_img_count=args.class_img_count,
    )
    dl = DataLoader(ds, 1)
    with torch.no_grad():
        clas_prompt_enc = model.get_learned_conditioning(class_prompt)
        inst_prompt_enc = model.get_learned_conditioning(instance_prompt)

    ckpt_dir = os.path.join(output_dir, "ckpts")
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    epochs = args.epochs
    tb_writer = SummaryWriter(log_dir)
    print("Starting Dreambooth training...")
    data_len = len(dl)
    for epoch in range(epochs):
        with tqdm(
            total=data_len,
            postfix={"loss": "undefined"},
            desc=f"EPOCH {epoch+1}/{epochs}",
        ) as pbar:
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
                tb_writer.add_scalar("Loss", loss, i + epoch * data_len)

                if (
                    args.save_every != -1
                    and (epoch * data_len + i) % args.save_every == 0
                    and i != data_len - 1
                    and (i != 0)
                ):
                    torch.save(
                        {
                            "state_dict": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "epoch": epoch,
                        },
                        os.path.join(ckpt_dir, f"e{epoch}i{i}.ckpt"),
                    )

                pbar.update()

            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                },
                os.path.join(ckpt_dir, f"e{epoch}.ckpt"),
            )

    print("Training complete!")
