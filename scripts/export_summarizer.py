"""Export summarizer LLM to CoreML via ANEMLL for on-device iOS inference.

Uses ANEMLL to convert Qwen3-4B (or other supported models) to CoreML
optimized for Apple Neural Engine. ANEMLL handles chunking, quantization
(LUT4/LUT6), and ANE-specific optimizations.

IMPORTANT: This script requires macOS with Apple Silicon (ANEMLL requirement).
           It will NOT work on Linux.

Usage:
    uv run python scripts/export_summarizer.py                              # Default: Qwen3.5-4B, ctx2048
    uv run python scripts/export_summarizer.py --model Qwen/Qwen3-4B       # Qwen3 alternative
    uv run python scripts/export_summarizer.py --context 4096               # Longer context
    uv run python scripts/export_summarizer.py --fp16                       # No quantization (larger)

Output:
    exports/Summarizer_{model_name}/
    ├── meta.yaml               # ANEMLL model metadata
    ├── embeddings/             # Embedding layer CoreML
    ├── lm_head/                # LM head CoreML
    └── ffn_*/                  # FFN chunks CoreML

Target consumer: brevox-ios (on-device summarization)
Prerequisites: ANEMLL installed (see https://github.com/Anemll/Anemll)
"""

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()

DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_CONTEXT = 2048  # Transcripts ~500-2000 tokens + JSON output ~200 tokens
DEFAULT_BATCH = 64


def check_prerequisites():
    """Verify we're on macOS with Apple Silicon and ANEMLL is available."""
    if platform.system() != "Darwin":
        console.print("[bold red]Error: ANEMLL requires macOS with Apple Silicon.[/bold red]")
        console.print("This script cannot run on Linux. Export on your Mac instead.")
        console.print("\nSee: https://github.com/Anemll/Anemll")
        sys.exit(1)

    if platform.machine() not in ("arm64", "aarch64"):
        console.print("[bold red]Error: ANEMLL requires Apple Silicon (arm64).[/bold red]")
        sys.exit(1)


def find_anemll() -> Path | None:
    """Find ANEMLL installation."""
    # Check if convert_model.sh is in PATH or common locations
    candidates = [
        Path.home() / "Anemll" / "anemll" / "utils" / "convert_model.sh",
        Path.home() / "Projects" / "Anemll" / "anemll" / "utils" / "convert_model.sh",
        Path(".models") / "Anemll" / "anemll" / "utils" / "convert_model.sh",
    ]

    for path in candidates:
        if path.exists():
            return path

    # Try which
    result = subprocess.run(["which", "anemll"], capture_output=True, text=True)
    if result.returncode == 0:
        return Path(result.stdout.strip())

    return None


def install_anemll() -> Path:
    """Clone and setup ANEMLL."""
    anemll_dir = Path(".models") / "Anemll"
    if not anemll_dir.exists():
        console.print("[bold]Cloning ANEMLL...[/bold]")
        anemll_dir.parent.mkdir(exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth=1",
             "https://github.com/Anemll/Anemll.git",
             str(anemll_dir)],
            check=True,
        )
    convert_script = anemll_dir / "anemll" / "utils" / "convert_model.sh"
    if not convert_script.exists():
        console.print(f"[bold red]convert_model.sh not found at {convert_script}[/bold red]")
        sys.exit(1)
    return convert_script


def model_name_short(model_id: str) -> str:
    """Extract short name from HuggingFace model ID."""
    return model_id.split("/")[-1].replace("-", "_")


def main():
    parser = argparse.ArgumentParser(
        description="Export summarizer LLM to CoreML via ANEMLL"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HuggingFace model ID. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=DEFAULT_CONTEXT,
        help=f"Context length (tokens). Default: {DEFAULT_CONTEXT}",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=DEFAULT_BATCH,
        help=f"Batch size for prefill. Default: {DEFAULT_BATCH}",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Skip quantization (FP16 only). Larger but higher quality.",
    )
    parser.add_argument(
        "--chunks",
        type=int,
        default=1,
        help="Number of chunks (1 = monolithic). Default: 1",
    )
    parser.add_argument(
        "--output-dir",
        default="exports",
        help="Output directory. Default: exports/",
    )
    args = parser.parse_args()

    short_name = model_name_short(args.model)
    output_dir = Path(args.output_dir) / f"Summarizer_{short_name}"

    quant_label = "FP16" if args.fp16 else "LUT4+LUT6"

    # Print config
    table = Table(title="Summarizer Export Config")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Model", args.model)
    table.add_row("Context Length", f"{args.context} tokens")
    table.add_row("Batch Size", str(args.batch))
    table.add_row("Quantization", quant_label)
    table.add_row("Chunks", str(args.chunks))
    table.add_row("Output Dir", str(output_dir))
    console.print(table)

    # Step 1: Check prerequisites
    console.print("\n[bold]Step 1/3: Checking prerequisites...[/bold]")
    check_prerequisites()
    console.print("  [green]macOS + Apple Silicon detected[/green]")

    # Step 2: Find or install ANEMLL
    console.print("[bold]Step 2/3: Locating ANEMLL...[/bold]")
    convert_script = find_anemll()
    if convert_script is None:
        console.print("  ANEMLL not found, installing...")
        convert_script = install_anemll()
    console.print(f"  [green]Found: {convert_script}[/green]")

    # Step 3: Run conversion
    console.print(f"[bold]Step 3/3: Converting {args.model} to CoreML...[/bold]")
    console.print("  This may take 10-30 minutes depending on model size and RAM.")

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(convert_script),
        "--model", args.model,
        "--output", str(output_dir),
        "--context", str(args.context),
        "--batch", str(args.batch),
        "--chunk", str(args.chunks),
        "--argmax",
    ]

    if not args.fp16:
        # LUT4 for FFN layers, LUT6 for LM head (ANEMLL recommended defaults)
        cmd.extend(["--lut2", "4", "--lut3", "6"])

    console.print(f"  Command: [dim]{' '.join(cmd)}[/dim]")

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        console.print(f"\n[bold red]Conversion failed (exit code {result.returncode})[/bold red]")
        console.print("Check ANEMLL logs above for details.")
        sys.exit(1)

    # Summary
    console.print(f"\n[bold green]Export complete![/bold green]")

    info_table = Table(title="Summarizer Export Summary")
    info_table.add_column("Property", style="cyan")
    info_table.add_column("Value", style="green")
    info_table.add_row("Model", args.model)
    info_table.add_row("Context", f"{args.context} tokens")
    info_table.add_row("Quantization", quant_label)
    info_table.add_row("Output", str(output_dir))

    # Calculate total size
    if output_dir.exists():
        total_size = sum(
            f.stat().st_size for f in output_dir.rglob("*") if f.is_file()
        )
        info_table.add_row("Total Size", f"{total_size / 1024 / 1024:.1f} MB")

    console.print(info_table)

    console.print(f"\n[bold]Output directory:[/bold] {output_dir}/")
    console.print("  meta.yaml → ANEMLL model config for Swift inference")
    console.print("  CoreML models → load with ANEMLL Swift runtime")

    console.print("\n[bold]Next step:[/bold] Copy to Brevox:")
    console.print(f"  cp -r {output_dir} ~/Developer/apps/brevox-ios/Resources/MLModels/")

    console.print("\n[bold]Test locally:[/bold]")
    anemll_dir = convert_script.parent.parent.parent
    console.print(f"  python {anemll_dir}/tests/chat.py --meta {output_dir}/meta.yaml")


if __name__ == "__main__":
    main()
