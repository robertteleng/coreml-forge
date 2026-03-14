"""Export Face Landmarks (MediaPipe-compatible 468 points) to CoreML for ANE inference.

Uses a PyTorch reimplementation of MediaPipe FaceMesh from zmurez/MediaPipePyTorch.
Output landmarks match MediaPipe's 468-point face mesh indices (drop-in replacement).

Usage:
    uv run python scripts/export_face_landmarks.py                  # Default: FP16, ANE
    uv run python scripts/export_face_landmarks.py --int8           # INT8 quantization (2x faster on ANE)
    uv run python scripts/export_face_landmarks.py --no-half        # FP32 instead of FP16

Output: exports/FaceLandmarks.mlpackage

Target consumer: flow-coach-ios (replaces MediaPipe FaceLandmarker GPU → CoreML ANE)
"""

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()

REPO_URL = "https://github.com/zmurez/MediaPipePyTorch.git"
REPO_DIR = Path(".models/MediaPipePyTorch")
WEIGHTS_FILE = "blazeface_landmark.pth"
INPUT_SIZE = 192
NUM_LANDMARKS = 468


def ensure_repo() -> Path:
    """Clone MediaPipePyTorch repo if not present."""
    REPO_DIR.parent.mkdir(exist_ok=True)
    if not REPO_DIR.exists():
        console.print("[bold]Cloning MediaPipePyTorch repository...[/bold]")
        import subprocess

        subprocess.run(
            ["git", "clone", "--depth=1", REPO_URL, str(REPO_DIR)],
            check=True,
        )
    return REPO_DIR


def load_model(repo_dir: Path):
    """Load BlazeFaceLandmark model with pretrained weights."""
    import torch

    # Add repo to path for imports
    sys.path.insert(0, str(repo_dir))
    from blazeface_landmark import BlazeFaceLandmark

    model = BlazeFaceLandmark()
    weights_path = repo_dir / WEIGHTS_FILE
    if not weights_path.exists():
        console.print(f"[bold red]Weights not found at {weights_path}[/bold red]")
        console.print("Expected weights file in the cloned repo.")
        raise FileNotFoundError(f"Missing {weights_path}")

    model.load_state_dict(torch.load(str(weights_path), map_location="cpu"))
    model.eval()
    return model


def create_traceable_model(model):
    """Create a traceable nn.Module from BlazeFaceLandmark.

    The original forward() has `if x.shape[0] == 0: return ...` which is
    data-dependent control flow and breaks torch.jit.trace. We rewrite
    forward() using the original's backbone layers directly.

    Original forward:
        x = F.pad(x, (0,1,0,1), "constant", 0)
        x = backbone1(x)
        landmarks = backbone2a(x).view(-1, 468, 3) / 192
        flag = backbone2b(x).sigmoid().view(-1)
        return flag, landmarks
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class TraceableFaceLandmark(nn.Module):
        def __init__(self, src):
            super().__init__()
            self.backbone1 = src.backbone1
            self.backbone2a = src.backbone2a
            self.backbone2b = src.backbone2b

        def forward(self, x):
            x = F.pad(x, (0, 1, 0, 1), "constant", 0)
            x = self.backbone1(x)
            landmarks = self.backbone2a(x).reshape(1, NUM_LANDMARKS, 3) / INPUT_SIZE
            confidence = self.backbone2b(x).sigmoid().reshape(1)
            return landmarks, confidence

    traceable = TraceableFaceLandmark(model)
    traceable.eval()

    # Verify output shapes
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    with torch.no_grad():
        landmarks, confidence = traceable(dummy)
    console.print(f"  Landmarks shape: [cyan]{list(landmarks.shape)}[/cyan]")
    console.print(f"  Confidence shape: [cyan]{list(confidence.shape)}[/cyan]")

    return traceable


def main():
    parser = argparse.ArgumentParser(
        description="Export Face Landmarks (468-point) to CoreML for ANE"
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Apply INT8 quantization (2x faster on ANE, slight accuracy loss)",
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    precision_label = "INT8" if args.int8 else ("FP32" if args.no_half else "FP16")

    # Print config
    table = Table(title="Face Landmarks Export Config")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Model", "BlazeFaceLandmark (zmurez/MediaPipePyTorch)")
    table.add_row("Input Size", f"{INPUT_SIZE}x{INPUT_SIZE}")
    table.add_row("Landmarks", str(NUM_LANDMARKS))
    table.add_row("Precision", precision_label)
    table.add_row("Compute", "CPU + Neural Engine")
    table.add_row("Output Dir", str(output_dir))
    console.print(table)

    # Step 1: Clone repo
    console.print("\n[bold]Step 1/5: Preparing model source...[/bold]")
    repo_dir = ensure_repo()

    # Step 2: Load model
    console.print("[bold]Step 2/5: Loading PyTorch model...[/bold]")
    model = load_model(repo_dir)
    console.print(f"  [green]Model loaded with {NUM_LANDMARKS} landmarks[/green]")

    # Step 3: Create traceable version and trace
    console.print("[bold]Step 3/5: Tracing model with JIT...[/bold]")
    import torch

    traceable = create_traceable_model(model)

    dummy_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    with torch.no_grad():
        traced = torch.jit.trace(traceable, dummy_input)

    console.print("  [green]Trace successful[/green]")

    # Step 4: Convert to CoreML
    console.print("[bold]Step 4/5: Converting to CoreML...[/bold]")
    import coremltools as ct

    precision = ct.precision.FLOAT32 if args.no_half else ct.precision.FLOAT16

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.ImageType(
                name="image",
                shape=(1, 3, INPUT_SIZE, INPUT_SIZE),
                scale=1 / 255.0,  # BlazeFaceLandmark expects [0, 1] range
            )
        ],
        outputs=[
            ct.TensorType(name="landmarks"),
            ct.TensorType(name="confidence"),
        ],
        compute_precision=precision,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
    )

    # Add model metadata
    mlmodel.author = "coreml-forge"
    mlmodel.short_description = (
        f"Face Landmarks ({NUM_LANDMARKS} points, MediaPipe-compatible). "
        f"Input: {INPUT_SIZE}x{INPUT_SIZE} RGB. "
        f"Output: [{NUM_LANDMARKS} x (x, y, z)] normalized coords + confidence."
    )
    mlmodel.input_description["image"] = (
        f"Face crop, {INPUT_SIZE}x{INPUT_SIZE} RGB"
    )
    mlmodel.output_description["landmarks"] = (
        f"{NUM_LANDMARKS} face landmarks, each (x, y, z) normalized [0,1]"
    )
    mlmodel.output_description["confidence"] = "Face detection confidence [0,1]"

    # INT8 quantization
    if args.int8:
        console.print("  [bold]Applying INT8 quantization...[/bold]")
        from coremltools.optimize.coreml import (
            OptimizationConfig,
            OpLinearQuantizerConfig,
            linear_quantize_weights,
        )

        op_config = OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
        config = OptimizationConfig(global_config=op_config)
        mlmodel = linear_quantize_weights(mlmodel, config=config)
        console.print("  [green]INT8 quantization applied[/green]")

    # Step 5: Save
    suffix = "_int8" if args.int8 else ""
    output_name = f"FaceLandmarks{suffix}.mlpackage"
    output_path = output_dir / output_name

    console.print(f"[bold]Step 5/5: Saving to {output_path}...[/bold]")

    if output_path.exists():
        import shutil

        shutil.rmtree(output_path)

    mlmodel.save(str(output_path))

    # Summary
    console.print(f"\n[bold green]Export complete![/bold green]")

    info_table = Table(title="Model Info")
    info_table.add_column("Property", style="cyan")
    info_table.add_column("Value", style="green")
    info_table.add_row("Model", "BlazeFaceLandmark")
    info_table.add_row("Landmarks", f"{NUM_LANDMARKS} (MediaPipe-compatible)")
    info_table.add_row("Input", f"{INPUT_SIZE}x{INPUT_SIZE} RGB, [0-255] → [0-1]")
    info_table.add_row("Output", f"[{NUM_LANDMARKS}, 3] (x,y,z) + confidence")
    info_table.add_row("Precision", precision_label)
    info_table.add_row("Compute", "CPU + Neural Engine")
    info_table.add_row("Key Indices", "chin=152, nose_tip=1, left_eye=33, right_eye=263")

    if output_path.is_dir():
        total_size = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file())
        info_table.add_row("Size", f"{total_size / 1024 / 1024:.1f} MB")

    console.print(info_table)

    console.print("\n[bold]Next step:[/bold] Copy to your Xcode project:")
    console.print(
        f"  cp -r {output_path} ~/Developer/FlowCoach/Resources/MLModels/"
    )

    console.print("\n[bold]Verify landmarks match MediaPipe:[/bold]")
    console.print("  - Chin: index 152")
    console.print("  - Nose tip: index 1")
    console.print("  - Left eye inner corner: index 33")
    console.print("  - Right eye inner corner: index 263")


if __name__ == "__main__":
    main()
