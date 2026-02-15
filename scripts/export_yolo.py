"""Export YOLO model to CoreML (.mlpackage) for iOS inference.

Usage:
    uv run python scripts/export_yolo.py                    # Default: yolo26s, 640x640, FP16
    uv run python scripts/export_yolo.py --model yolo11s    # Different model
    uv run python scripts/export_yolo.py --imgsz 320        # Smaller input (faster, less accurate)
    uv run python scripts/export_yolo.py --no-nms           # Without NMS (handle in Swift)
    uv run python scripts/export_yolo.py --no-half          # FP32 instead of FP16

Output: exports/<model_name>.mlpackage
"""

import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Export YOLO to CoreML")
    parser.add_argument(
        "--model",
        default="yolo26s",
        help="YOLO model name (e.g., yolo26s, yolo26n, yolo11s, yolov8s). Default: yolo26s",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size. Default: 640. Use 320 for faster inference.",
    )
    parser.add_argument(
        "--no-nms",
        action="store_true",
        help="Disable NMS in the model (handle NMS in Swift instead)",
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

    # Print config
    table = Table(title="YOLO Export Config")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Model", args.model)
    table.add_row("Input Size", f"{args.imgsz}x{args.imgsz}")
    table.add_row("NMS", "Included" if not args.no_nms else "Disabled")
    table.add_row("Precision", "FP32" if args.no_half else "FP16")
    table.add_row("Output Dir", str(output_dir))
    console.print(table)

    # Import here so --help is fast
    from ultralytics import YOLO

    # Load model (downloads weights automatically)
    console.print(f"\n[bold]Loading {args.model}...[/bold]")
    model_name = args.model if args.model.endswith(".pt") else f"{args.model}.pt"
    model = YOLO(model_name)

    # Export to CoreML
    console.print("[bold]Exporting to CoreML...[/bold]")
    export_path = model.export(
        format="coreml",
        nms=not args.no_nms,
        imgsz=args.imgsz,
        half=not args.no_half,
    )

    # Move to output directory
    export_file = Path(export_path)
    dest = output_dir / export_file.name
    if export_file != dest:
        if dest.exists():
            import shutil
            shutil.rmtree(dest)
        export_file.rename(dest)

    # Summary
    console.print(f"\n[bold green]Export complete![/bold green]")
    console.print(f"  Output: [cyan]{dest}[/cyan]")

    # Print model info
    info_table = Table(title="Model Info")
    info_table.add_column("Property", style="cyan")
    info_table.add_column("Value", style="green")
    info_table.add_row("Classes", str(len(model.names)))
    info_table.add_row("Input", f"{args.imgsz}x{args.imgsz}")
    info_table.add_row("NMS", "Built-in" if not args.no_nms else "External")

    # Check file size
    if dest.is_dir():
        total_size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
        info_table.add_row("Size", f"{total_size / 1024 / 1024:.1f} MB")

    console.print(info_table)

    console.print("\n[bold]Next step:[/bold] Copy to your Xcode project:")
    console.print(f"  cp -r {dest} ~/Developer/extreme/rayban-nav/RayBanNav/Resources/MLModels/")


if __name__ == "__main__":
    main()
