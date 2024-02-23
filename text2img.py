from argparse import ArgumentParser
from src.util.io import get_output_path, load_model_from_config
from src.util.diffusion import sample_to_dir
from omegaconf import OmegaConf
from src.samplers import PLMSSampler
import os
import ast


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "-p", "--prompt", type=str, required=True, help="Prompt to convert to image"
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=None,
        help="Generation batch size",
    )
    parser.add_argument(
        "-v",
        "--variations",
        type=int,
        default=1,
        help="Number of variations to be generated",
    )
    parser.add_argument(
        "-o", "--output-dir", default="out/txt2img", help="Directory to save outputs"
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

    sample_to_dir(
        sampler,
        args.prompt,
        samples_dir,
        args.hw,
        args.ddim_steps,
        args.scale,
        args.ddim_eta,
        args.batch_size if args.batch_size is not None else args.variations,
        args.variations,
        args.precision,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
