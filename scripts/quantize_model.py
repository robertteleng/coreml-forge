"""Post-training quantization for CoreML models using INT8.

Applies INT8 quantization to reduce model size and improve Neural Engine performance.
Supports W8A8 (weights + activations INT8) for maximum ANE throughput on A17 Pro/M4.

Usage:
    # Linear quantization (weights only, fast)
    uv run python scripts/quantize_model.py exports/DepthAnythingV2_vits.mlpackage

    # W8A8 quantization (weights + activations, requires calibration data)
    uv run python scripts/quantize_model.py exports/DepthAnythingV2_vits.mlpackage --mode w8a8

    # Specify output path
    uv run python scripts/quantize_model.py model.mlpackage --output model_int8.mlpackage

Performance gains on A17 Pro / M4:
    - FP16 → INT8 weights: ~2x smaller, ~1.3x faster
    - FP16 → W8A8: ~4x smaller, ~2x faster (ANE int8-int8 compute path)

References:
    - https://apple.github.io/coremltools/docs-guides/source/opt-quantization-overview.html
    - https://apple.github.io/coremltools/docs-guides/source/opt-workflow.html
"""

import argparse
import shutil
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


def quantize_linear(model_path: Path, output_path: Path) -> dict:
    """Apply linear INT8 quantization to weights only (no calibration needed).

    This is the simplest form of quantization. Weights are quantized to INT8
    and dequantized at runtime. Faster than W8A8 but less accurate.
    """
    import coremltools as ct
    from coremltools.optimize.coreml import (
        OpLinearQuantizerConfig,
        OptimizationConfig,
        linear_quantize_weights,
    )

    console.print("[bold]Loading model...[/bold]")
    model = ct.models.MLModel(str(model_path))

    console.print("[bold]Applying linear INT8 quantization...[/bold]")

    # Configure INT8 quantization for all ops
    op_config = OpLinearQuantizerConfig(
        mode="linear_symmetric",  # Symmetric quantization (better for ANE)
        dtype="int8",
        granularity="per_tensor",  # per_tensor is faster, per_channel more accurate
    )
    config = OptimizationConfig(global_config=op_config)

    quantized_model = linear_quantize_weights(model, config=config)

    console.print(f"[bold]Saving to {output_path}...[/bold]")
    if output_path.exists():
        shutil.rmtree(output_path)
    quantized_model.save(str(output_path))

    return {
        "method": "linear_symmetric_int8",
        "granularity": "per_tensor",
        "calibration": "none",
    }


def quantize_w8a8(
    model_path: Path,
    output_path: Path,
    calibration_dir: Path | None = None,
    num_samples: int = 100,
) -> dict:
    """Apply W8A8 quantization (weights AND activations to INT8).

    This enables the ANE's optimized int8-int8 compute path on A17 Pro/M4.
    Requires calibration data for activation quantization.

    If no calibration_dir is provided, uses random calibration data (less accurate
    but still effective for many models).
    """
    import coremltools as ct
    from coremltools.optimize.coreml import (
        OptimizationConfig,
        OpLinearQuantizerConfig,
        linear_quantize_weights,
    )
    import numpy as np
    from PIL import Image

    console.print("[bold]Loading model...[/bold]")
    model = ct.models.MLModel(str(model_path))

    # Get input spec to determine calibration data shape
    spec = model.get_spec()
    input_desc = spec.description.input[0]
    input_name = input_desc.name

    # Determine input shape from model spec
    if input_desc.type.HasField("imageType"):
        img_type = input_desc.type.imageType
        height = img_type.height
        width = img_type.width
        # ImageType expects PIL images or numpy arrays in specific format
        is_image_input = True
        console.print(f"  Input: {input_name} (image {width}x{height})")
    elif input_desc.type.HasField("multiArrayType"):
        shape = list(input_desc.type.multiArrayType.shape)
        is_image_input = False
        console.print(f"  Input: {input_name} (array {shape})")
    else:
        console.print("[red]Unknown input type, falling back to linear quantization[/red]")
        return quantize_linear(model_path, output_path)

    # Generate calibration data
    console.print(f"[bold]Generating calibration data ({num_samples} samples)...[/bold]")

    def make_calibration_data():
        """Generator for calibration samples."""
        if calibration_dir and calibration_dir.exists():
            # Use real images from calibration directory
            image_files = list(calibration_dir.glob("*.jpg")) + list(calibration_dir.glob("*.png"))
            for img_path in image_files[:num_samples]:
                img = Image.open(img_path).convert("RGB")
                if is_image_input:
                    img = img.resize((width, height))
                    yield {input_name: img}
                else:
                    # Convert to numpy array matching expected shape
                    img = img.resize((shape[-1], shape[-2]))
                    arr = np.array(img).astype(np.float32) / 255.0
                    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
                    arr = np.expand_dims(arr, 0)  # Add batch dim
                    yield {input_name: arr}
        else:
            # Generate random calibration data
            console.print("  [yellow]No calibration images provided, using random data[/yellow]")
            for _ in range(num_samples):
                if is_image_input:
                    # Random RGB image
                    arr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
                    img = Image.fromarray(arr, "RGB")
                    yield {input_name: img}
                else:
                    # Random tensor matching shape
                    arr = np.random.rand(*shape).astype(np.float32)
                    yield {input_name: arr}

    # Configure W8A8 quantization
    console.print("[bold]Applying W8A8 quantization...[/bold]")

    # First quantize weights
    weight_config = OpLinearQuantizerConfig(
        mode="linear_symmetric",
        dtype="int8",
        granularity="per_channel",  # per_channel for better accuracy
    )

    config = OptimizationConfig(
        global_config=weight_config,
    )

    # Step 1: Quantize weights
    console.print("  Step 1/2: Quantizing weights...")
    quantized_model = linear_quantize_weights(model, config=config)

    # Step 2: Quantize activations using calibration data
    console.print("  Step 2/2: Quantizing activations with calibration...")
    try:
        from coremltools.optimize.coreml import (
            OpActivationLinearQuantizerConfig,
            experimental,
        )

        activation_config = OpActivationLinearQuantizerConfig(
            mode="linear_symmetric",
        )
        quantized_model = experimental.linear_quantize_activations(
            quantized_model,
            make_calibration_data(),
            activation_config,
        )
    except (ImportError, AttributeError) as e:
        console.print(f"  [yellow]Activation quantization not available in this coremltools version: {e}[/yellow]")
        console.print("  [yellow]Falling back to weight-only INT8 quantization[/yellow]")
    except Exception as e:
        console.print(f"  [yellow]Activation quantization failed: {e}[/yellow]")
        console.print("  [yellow]Falling back to weight-only INT8 quantization[/yellow]")

    console.print(f"[bold]Saving to {output_path}...[/bold]")
    if output_path.exists():
        shutil.rmtree(output_path)
    quantized_model.save(str(output_path))

    return {
        "method": "w8a8",
        "weight_granularity": "per_channel",
        "activation_granularity": "per_tensor",
        "calibration_samples": num_samples,
        "calibration_source": str(calibration_dir) if calibration_dir else "random",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Quantize CoreML model to INT8",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "model",
        type=Path,
        help="Path to CoreML model (.mlpackage)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output path (default: model_int8.mlpackage)",
    )
    parser.add_argument(
        "--mode",
        choices=["linear", "w8a8"],
        default="linear",
        help="Quantization mode: 'linear' (weights only) or 'w8a8' (weights + activations). Default: linear",
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=None,
        help="Directory with calibration images (for w8a8 mode)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of calibration samples (for w8a8 mode). Default: 100",
    )
    args = parser.parse_args()

    model_path = args.model
    if not model_path.exists():
        console.print(f"[red]Model not found: {model_path}[/red]")
        return 1

    # Default output path
    if args.output:
        output_path = args.output
    else:
        suffix = "_int8" if args.mode == "linear" else "_w8a8"
        output_path = model_path.with_name(model_path.stem + suffix + ".mlpackage")

    # Print config
    table = Table(title="Quantization Config")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Input Model", str(model_path))
    table.add_row("Output Model", str(output_path))
    table.add_row("Mode", args.mode)
    if args.mode == "w8a8":
        table.add_row("Calibration Dir", str(args.calibration_dir) if args.calibration_dir else "random")
        table.add_row("Calibration Samples", str(args.num_samples))
    console.print(table)

    # Run quantization
    if args.mode == "linear":
        result = quantize_linear(model_path, output_path)
    else:
        result = quantize_w8a8(model_path, output_path, args.calibration_dir, args.num_samples)

    # Summary
    console.print(f"\n[bold green]Quantization complete![/bold green]")

    # Compare sizes
    def get_size(path: Path) -> int:
        if path.is_dir():
            return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        return path.stat().st_size

    original_size = get_size(model_path)
    quantized_size = get_size(output_path)
    reduction = (1 - quantized_size / original_size) * 100

    info_table = Table(title="Results")
    info_table.add_column("Property", style="cyan")
    info_table.add_column("Value", style="green")
    info_table.add_row("Method", result["method"])
    info_table.add_row("Original Size", f"{original_size / 1024 / 1024:.1f} MB")
    info_table.add_row("Quantized Size", f"{quantized_size / 1024 / 1024:.1f} MB")
    info_table.add_row("Size Reduction", f"{reduction:.1f}%")
    info_table.add_row("Output", str(output_path))
    console.print(info_table)

    console.print("\n[bold]Next step:[/bold] Copy to your Xcode project:")
    console.print(f"  cp -r {output_path} ~/Developer/extreme/rayban-nav/RayBanNav/Resources/MLModels/")

    return 0


if __name__ == "__main__":
    exit(main())
