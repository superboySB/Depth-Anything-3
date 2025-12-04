#!/usr/bin/env python3
"""
Minimal sanity-check runner that mirrors the workflow in notebooks/da3.ipynb.
Loads the pretrained model, runs inference on sample images, and saves depth
visualizations plus optional exports so we can confirm the environment works.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
from PIL import Image

import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from depth_anything_3.api import DepthAnything3  # noqa: E402
from depth_anything_3.utils.visualize import visualize_depth  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick DA3 depth+pose smoke test.")
    parser.add_argument(
        "--images",
        nargs="+",
        default=[
            "assets/examples/SOH/000.png",
            "assets/examples/SOH/010.png",
        ],
        help="Paths to images to feed into the model.",
    )
    parser.add_argument(
        "--model-name",
        default="depth-anything/DA3NESTED-GIANT-LARGE",
        help="Model repo or local directory passed to DepthAnything3.from_pretrained.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device string understood by torch (cuda/cpu).",
    )
    parser.add_argument(
        "--process-res",
        type=int,
        default=504,
        help="Base processing resolution (see docs/API.md).",
    )
    parser.add_argument(
        "--process-res-method",
        default="upper_bound_resize",
        choices=["upper_bound_resize", "lower_bound_resize"],
        help="Image resize strategy applied before inference.",
    )
    parser.add_argument(
        "--export-dir",
        default=str(Path("workspace") / "test_output"),
        help="Optional directory for export artifacts (glb/mini_npz/etc).",
    )
    parser.add_argument(
        "--export-format",
        default="glb",
        help="Export formats separated by '-', or 'none' to skip exporting.",
    )
    parser.add_argument(
        "--feat-layers",
        default="",
        help="Comma-separated feature layers for feat_vis export ('' means skip).",
    )
    parser.add_argument(
        "--conf-thresh-percentile",
        type=float,
        default=40.0,
        help="GLB confidence percentile (see docs/CLI.md).",
    )
    parser.add_argument(
        "--num-max-points",
        type=int,
        default=1_000_000,
        help="GLB max number of points.",
    )
    parser.add_argument(
        "--show-cameras",
        action="store_true",
        help="Include camera wireframes in GLB export.",
    )
    parser.add_argument(
        "--output-prefix",
        default="depth_vis",
        help="Output filename prefix inside export directory.",
    )
    return parser.parse_args()


def ensure_paths_exist(paths: List[str]) -> None:
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            f"Could not find the following image paths:\n" + "\n".join(missing)
        )


def save_depth_visualizations(prediction, output_dir: Path, prefix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, depth in enumerate(prediction.depth):
        vis = visualize_depth(depth, cmap="Spectral")
        image = Image.fromarray(vis)
        image_path = output_dir / f"{prefix}_{idx:03d}.png"
        image.save(image_path)

        if prediction.processed_images is not None:
            rgb = Image.fromarray(prediction.processed_images[idx])
            rgb.save(output_dir / f"{prefix}_{idx:03d}_input.png")


def main() -> None:
    args = parse_args()
    ensure_paths_exist(args.images)

    export_format = None if args.export_format.lower() == "none" else args.export_format
    export_feat_layers = (
        [int(idx) for idx in args.feat_layers.split(",") if idx.strip()]
        if args.feat_layers
        else []
    )

    device = torch.device(args.device)
    print(f"Loading model {args.model_name} on {device}...")
    model = DepthAnything3.from_pretrained(args.model_name).to(device)
    model.eval()

    print("Running inference ...")
    prediction = model.inference(
        image=args.images,
        process_res=args.process_res,
        process_res_method=args.process_res_method,
        export_dir=args.export_dir if export_format else None,
        export_format=export_format or "mini_npz",
        export_feat_layers=export_feat_layers,
        conf_thresh_percentile=args.conf_thresh_percentile,
        num_max_points=args.num_max_points,
        show_cameras=args.show_cameras,
    )

    print(f"Depth tensor shape: {prediction.depth.shape}")
    if prediction.extrinsics is not None:
        print(f"Extrinsics shape: {prediction.extrinsics.shape}")
    if prediction.intrinsics is not None:
        print(f"Intrinsics shape: {prediction.intrinsics.shape}")
    if prediction.aux:
        print(f"Aux keys: {list(prediction.aux.keys())}")

    save_depth_visualizations(prediction, Path(args.export_dir), args.output_prefix)
    print(f"Saved depth visualization(s) to {args.export_dir}")


if __name__ == "__main__":
    main()
