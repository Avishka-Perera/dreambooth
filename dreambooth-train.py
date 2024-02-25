from argparse import ArgumentParser
import ast
import os
from src.util.diffusion import txt2img
from src.util.io import get_output_path
import torch
from torch import multiprocessing as mp


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-d",
        "--devices",
        type=ast.literal_eval,
        nargs="+",
        help="CUDA Devices that must be used for training",
        default=list(range(torch.cuda.device_count())),
    )
    parser.add_argument(
        "-i",
        "--instance-prompt",
        type=str,
        required=True,
        help="Prompt to convert to image",
    )
    parser.add_argument(
        "-c",
        "--class-prompt",
        type=str,
        required=True,
        help="Prompt to convert to image",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        required=True,
        help="Generation batch size",
    )
    parser.add_argument(
        "-v",
        "--variations",
        type=int,
        default=1,
        help="Number of variations to be generated from the class prompt",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="out/dreambooth-train",
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
        "--lambda",
        type=float,
        default=1.0,
        help="",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    output_dir = get_output_path(args.output_dir, "run")

    # generate from class prompt
    class_img_dir = os.path.join(output_dir, "class-imgs")
    print(
        f"Exporting class images for the prompt '{args.class_prompt}', to directory {class_img_dir}"
    )
    world_size = len(args.devices)
    mp.spawn(
        txt2img,
        (
            world_size,
            args.class_prompt,
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
