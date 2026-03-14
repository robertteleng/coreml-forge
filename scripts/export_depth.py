"""Export Depth Anything V2 to CoreML (.mlpackage) for iOS depth estimation.

Usage:
    uv run python scripts/export_depth.py                      # Default: vits, 518x518, FP16
    uv run python scripts/export_depth.py --variant vitb        # Base model (larger, more accurate)
    uv run python scripts/export_depth.py --imgsz 256           # Smaller input (faster)

Output: exports/DepthAnythingV2_<variant>.mlpackage

Prerequisites:
    The script clones the Depth Anything V2 repo and downloads weights from HuggingFace automatically.
"""

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()

VARIANTS = {
    "vits": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
        "hf_id": "depth-anything/Depth-Anything-V2-Small",
        "weights_file": "depth_anything_v2_vits.pth",
    },
    "vitb": {
        "encoder": "vitb",
        "features": 128,
        "out_channels": [96, 192, 384, 768],
        "hf_id": "depth-anything/Depth-Anything-V2-Base",
        "weights_file": "depth_anything_v2_vitb.pth",
    },
    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
        "hf_id": "depth-anything/Depth-Anything-V2-Large",
        "weights_file": "depth_anything_v2_vitl.pth",
    },
}


def ensure_depth_anything_repo() -> Path:
    """Clone Depth Anything V2 repo if not present."""
    models_dir = Path(".models")
    models_dir.mkdir(exist_ok=True)
    repo_dir = models_dir / "Depth-Anything-V2"
    if not repo_dir.exists():
        console.print("[bold]Cloning Depth Anything V2 repository...[/bold]")
        import subprocess

        subprocess.run(
            ["git", "clone", "--depth=1",
             "https://github.com/DepthAnything/Depth-Anything-V2.git",
             str(repo_dir)],
            check=True,
        )
    return repo_dir


def download_weights(variant: dict) -> Path:
    """Download model weights from HuggingFace."""
    weights_path = Path(".models") / variant["weights_file"]
    if not weights_path.exists():
        console.print(f"[bold]Downloading weights from {variant['hf_id']}...[/bold]")
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            repo_id=variant["hf_id"],
            filename=variant["weights_file"],
            local_dir=".models",
        )
    return weights_path


def main():
    parser = argparse.ArgumentParser(description="Export Depth Anything V2 to CoreML")
    parser.add_argument(
        "--variant",
        default="vits",
        choices=["vits", "vitb", "vitl"],
        help="Model variant: vits (Small, fast), vitb (Base, balanced), vitl (Large, accurate). Default: vits",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=518,
        help="Input image size. Default: 518. Use 256 for faster inference.",
    )
    parser.add_argument(
        "--no-half",
        action="store_true",
        help="Export as FP32 instead of FP16",
    )
    parser.add_argument(
        "--output-dir",
        default="exports",
        help="Output directory. Default: exports/",
    )
    args = parser.parse_args()

    variant = VARIANTS[args.variant]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print config
    table = Table(title="Depth Anything V2 Export Config")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Variant", f"{args.variant} ({variant['encoder']})")
    table.add_row("Input Size", f"{args.imgsz}x{args.imgsz}")
    table.add_row("Precision", "FP32" if args.no_half else "FP16")
    table.add_row("HuggingFace", variant["hf_id"])
    table.add_row("Output Dir", str(output_dir))
    console.print(table)

    # Step 1: Clone repo
    repo_dir = ensure_depth_anything_repo()

    # Add repo to path so we can import the model
    sys.path.insert(0, str(repo_dir))

    # Step 2: Download weights
    weights_path = download_weights(variant)

    # Step 3: Load PyTorch model
    console.print(f"\n[bold]Loading {args.variant} model...[/bold]")

    import torch
    from depth_anything_v2.dpt import DepthAnythingV2

    model = DepthAnythingV2(
        encoder=variant["encoder"],
        features=variant["features"],
        out_channels=variant["out_channels"],
    )
    model.load_state_dict(torch.load(str(weights_path), map_location="cpu"))
    model.eval()

    # Step 4: Trace model
    console.print("[bold]Tracing model with JIT...[/bold]")
    dummy_input = torch.randn(1, 3, args.imgsz, args.imgsz)
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy_input)

    # Step 5: Convert to CoreML
    console.print("[bold]Converting to CoreML...[/bold]")
    import coremltools as ct

    precision = ct.precision.FLOAT32 if args.no_half else ct.precision.FLOAT16

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.ImageType(
                name="image",
                shape=(1, 3, args.imgsz, args.imgsz),
                scale=1 / 255.0,
            )
        ],
        compute_precision=precision,
        minimum_deployment_target=ct.target.iOS18,
    )

    # Step 6: Save
    output_name = f"DepthAnythingV2_{args.variant}.mlpackage"
    output_path = output_dir / output_name
    console.print(f"[bold]Saving to {output_path}...[/bold]")

    if output_path.exists():
        import shutil
        shutil.rmtree(output_path)

    mlmodel.save(str(output_path))

    # Summary
    console.print(f"\n[bold green]Export complete![/bold green]")
    console.print(f"  Output: [cyan]{output_path}[/cyan]")

    info_table = Table(title="Model Info")
    info_table.add_column("Property", style="cyan")
    info_table.add_column("Value", style="green")
    info_table.add_row("Variant", args.variant)
    info_table.add_row("Input", f"{args.imgsz}x{args.imgsz}")
    info_table.add_row("Output", "Depth map (inverse: higher = closer)")
    info_table.add_row("Precision", "FP32" if args.no_half else "FP16")

    if output_path.is_dir():
        total_size = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file())
        info_table.add_row("Size", f"{total_size / 1024 / 1024:.1f} MB")

    console.print(info_table)

    console.print("\n[bold]Next step:[/bold] Copy to your Xcode project:")
    console.print(f"  cp -r {output_path} ~/Developer/extreme/rayban-nav/RayBanNav/Resources/MLModels/")


if __name__ == "__main__":
    main()
