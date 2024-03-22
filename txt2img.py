from argparse import ArgumentParser
from src.util.diffusion import txt2img
import ast
from src.util.io import get_output_path, save_args


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "-p", "--prompt", type=str, required=True, help="Prompt to convert to image"
    )
    parser.add_argument(
        "-d", "--device", type=int, default=0, help="Device to run the process on"
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=None,
        help="Generation batch size",
    )
    parser.add_argument(
        "-c",
        "--ckpt-path",
        type=str,
        default="weights/model.ckpt",
        help="Checkpoints path to the model",
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


if __name__ == "__main__":
    args = parse_args()

    output_dir = get_output_path(args.output_dir, "run")
    save_args(output_dir, args)

    txt2img(
        0,
        1,
        args.device,
        args.prompt,
        output_dir,
        args.hw,
        args.ddim_steps,
        args.scale,
        args.ddim_eta,
        args.batch_size if args.batch_size is not None else args.variations,
        args.variations,
        args.precision,
        args.ckpt_path,
    )
