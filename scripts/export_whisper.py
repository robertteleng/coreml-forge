"""Export Whisper encoder to CoreML + decoder weights/tokenizer for Swift implementation.

The encoder (mel → audio features) is exported as a CoreML model for ANE inference.
The decoder is NOT exported to CoreML (torch.jit.trace cannot handle its variable-length
autoregressive loop). Instead, decoder weights are saved as safetensors for loading in Swift,
and the decoder loop is implemented natively with Accelerate.framework.

Usage:
    uv run python scripts/export_whisper.py                          # Default: small, FP16
    uv run python scripts/export_whisper.py --variant tiny           # Tiny (~75 MB encoder)
    uv run python scripts/export_whisper.py --variant base           # Base (~140 MB encoder)
    uv run python scripts/export_whisper.py --no-half                # FP32

Output:
    exports/Whisper_{variant}/
    ├── WhisperEncoder_{variant}.mlpackage   # CoreML encoder (ANE)
    ├── decoder_weights.safetensors          # Decoder weights for Swift
    ├── tokenizer.json                       # BPE vocabulary + special tokens
    └── config.json                          # Model dimensions

Target consumer: brevox-ios (on-device transcription)
"""

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()

VARIANTS = ("tiny", "base", "small", "medium")
MEL_FRAMES = 3000  # 30s of audio at 16kHz → 3000 mel frames


def main():
    parser = argparse.ArgumentParser(
        description="Export Whisper encoder to CoreML + decoder assets for Swift"
    )
    parser.add_argument(
        "--variant",
        default="small",
        choices=VARIANTS,
        help="Whisper variant. Default: small",
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

    output_dir = Path(args.output_dir) / f"Whisper_{args.variant}"
    output_dir.mkdir(parents=True, exist_ok=True)

    precision_label = "FP32" if args.no_half else "FP16"

    # Print config
    table = Table(title="Whisper Export Config")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Variant", args.variant)
    table.add_row("Mel Input", f"80 × {MEL_FRAMES} (30s audio)")
    table.add_row("Encoder Precision", precision_label)
    table.add_row("Decoder", "Weights only (safetensors)")
    table.add_row("Output Dir", str(output_dir))
    console.print(table)

    # Step 1: Load Whisper model
    console.print(f"\n[bold]Step 1/5: Loading whisper-{args.variant}...[/bold]")
    import torch
    import whisper

    model = whisper.load_model(args.variant, device="cpu")
    model.eval()
    dims = model.dims
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    console.print(f"  [green]Loaded ({n_params:.0f}M parameters)[/green]")

    # Step 2: Trace and convert encoder
    console.print("[bold]Step 2/5: Exporting encoder to CoreML...[/bold]")

    encoder = model.encoder
    encoder.eval()

    mel_input = torch.randn(1, dims.n_mels, MEL_FRAMES)
    with torch.no_grad():
        traced_encoder = torch.jit.trace(encoder, mel_input)
        encoder_out = traced_encoder(mel_input)
    console.print(f"  Encoder output: [cyan]{list(encoder_out.shape)}[/cyan]")

    import coremltools as ct

    precision = ct.precision.FLOAT32 if args.no_half else ct.precision.FLOAT16

    ml_encoder = ct.convert(
        traced_encoder,
        inputs=[
            ct.TensorType(
                name="mel_spectrogram",
                shape=(1, dims.n_mels, MEL_FRAMES),
            )
        ],
        outputs=[
            ct.TensorType(name="audio_features"),
        ],
        compute_precision=precision,
        minimum_deployment_target=ct.target.iOS18,
    )
    ml_encoder.author = "coreml-forge"
    ml_encoder.short_description = (
        f"Whisper {args.variant} encoder. "
        f"Input: mel spectrogram [{dims.n_mels}, {MEL_FRAMES}]. "
        f"Output: audio features [{dims.n_audio_ctx}, {dims.n_audio_state}]."
    )

    encoder_path = output_dir / f"WhisperEncoder_{args.variant}.mlpackage"
    if encoder_path.exists():
        import shutil
        shutil.rmtree(encoder_path)
    ml_encoder.save(str(encoder_path))
    console.print(f"  [green]Saved {encoder_path}[/green]")

    # Step 3: Export decoder weights
    console.print("[bold]Step 3/5: Exporting decoder weights...[/bold]")
    from safetensors.torch import save_file

    decoder_state = model.decoder.state_dict()
    # Convert all tensors to float16 unless --no-half
    if not args.no_half:
        decoder_state = {
            k: v.half() if v.is_floating_point() else v
            for k, v in decoder_state.items()
        }

    weights_path = output_dir / "decoder_weights.safetensors"
    save_file(decoder_state, str(weights_path))
    n_tensors = len(decoder_state)
    console.print(f"  [green]Saved {n_tensors} tensors to {weights_path}[/green]")

    # Step 4: Export tokenizer
    console.print("[bold]Step 4/5: Exporting tokenizer...[/bold]")

    tokenizer = whisper.tokenizer.get_tokenizer(
        multilingual=(args.variant != "en"),
        num_languages=99,
    )

    # Extract BPE vocab from tiktoken encoding
    encoding = tokenizer.encoding
    vocab = {}
    for token_bytes, rank in encoding._mergeable_ranks.items():
        try:
            vocab[token_bytes.decode("utf-8")] = rank
        except UnicodeDecodeError:
            # Store non-UTF8 tokens as hex-escaped strings
            vocab[token_bytes.hex()] = rank

    special_tokens = {
        "sot": tokenizer.sot,
        "eot": tokenizer.eot,
        "sot_prev": tokenizer.sot_prev,
        "sot_lm": tokenizer.sot_lm,
        "no_speech": tokenizer.no_speech,
        "no_timestamps": tokenizer.no_timestamps,
        "timestamp_begin": tokenizer.timestamp_begin,
        "translate": tokenizer.translate,
        "transcribe": tokenizer.transcribe,
    }

    tokenizer_data = {
        "vocab": vocab,
        "special_tokens": special_tokens,
    }

    tokenizer_path = output_dir / "tokenizer.json"
    with open(tokenizer_path, "w") as f:
        json.dump(tokenizer_data, f, ensure_ascii=False)
    console.print(f"  [green]Saved {len(vocab)} vocab entries to {tokenizer_path}[/green]")

    # Step 5: Export config
    console.print("[bold]Step 5/5: Exporting config...[/bold]")

    config = {
        "variant": args.variant,
        "n_mels": dims.n_mels,
        "n_audio_ctx": dims.n_audio_ctx,
        "n_audio_state": dims.n_audio_state,
        "n_audio_head": dims.n_audio_head,
        "n_audio_layer": dims.n_audio_layer,
        "n_text_ctx": dims.n_text_ctx,
        "n_text_state": dims.n_text_state,
        "n_text_head": dims.n_text_head,
        "n_text_layer": dims.n_text_layer,
        "n_vocab": dims.n_vocab,
        "precision": precision_label,
        "mel_params": {
            "sample_rate": 16000,
            "n_fft": 400,
            "hop_length": 160,
            "n_mels": 80,
            "chunk_length_s": 30,
        },
    }

    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    console.print(f"  [green]Saved {config_path}[/green]")

    # Summary
    console.print(f"\n[bold green]Export complete![/bold green]")

    info_table = Table(title="Whisper Export Summary")
    info_table.add_column("Property", style="cyan")
    info_table.add_column("Value", style="green")
    info_table.add_row("Variant", args.variant)
    info_table.add_row("Parameters", f"{n_params:.0f}M")
    info_table.add_row("Encoder Input", f"mel [{dims.n_mels}, {MEL_FRAMES}]")
    info_table.add_row("Encoder Output", f"features [{dims.n_audio_ctx}, {dims.n_audio_state}]")
    info_table.add_row("Decoder", f"{n_tensors} weight tensors ({dims.n_text_layer} layers)")
    info_table.add_row("Vocab", f"{dims.n_vocab} tokens")
    info_table.add_row("Encoder Precision", precision_label)

    if encoder_path.is_dir():
        enc_size = sum(f.stat().st_size for f in encoder_path.rglob("*") if f.is_file())
        info_table.add_row("Encoder Size", f"{enc_size / 1024 / 1024:.1f} MB")
    if weights_path.exists():
        dec_size = weights_path.stat().st_size
        info_table.add_row("Decoder Weights", f"{dec_size / 1024 / 1024:.1f} MB")

    console.print(info_table)

    console.print(f"\n[bold]Output directory:[/bold] {output_dir}/")
    console.print("  WhisperEncoder → CoreML (ANE inference)")
    console.print("  decoder_weights.safetensors → load in Swift (Accelerate)")
    console.print("  tokenizer.json → BPE decode in Swift")
    console.print("  config.json → model dimensions for Swift decoder")

    console.print("\n[bold]Next step:[/bold] Copy to Brevox:")
    console.print(f"  cp -r {output_dir} ~/Developer/apps/brevox-ios/Resources/MLModels/")

    console.print("\n[bold]Note:[/bold] Mel spectrogram computed in Swift (Accelerate.framework)")
    console.print("  80 mel bins, 16kHz, hop=160, FFT=400, 30s chunks")


if __name__ == "__main__":
    main()
