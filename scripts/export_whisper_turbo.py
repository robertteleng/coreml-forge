"""Export Whisper large-v3-turbo encoder to CoreML + decoder weights for Swift.

Uses HuggingFace transformers (not openai-whisper) since large-v3-turbo is only
available there. The architecture is whisper-large with 128 mel bins and only
4 decoder layers (pruned from 32), giving near-large accuracy at much lower cost.

Usage:
    uv run python scripts/export_whisper_turbo.py                # Default: FP16
    uv run python scripts/export_whisper_turbo.py --no-half      # FP32

Output:
    exports/Whisper_large_v3_turbo/
    ├── WhisperEncoder_large_v3_turbo.mlpackage  # CoreML encoder (ANE)
    ├── decoder_weights.safetensors              # Decoder weights for Swift
    ├── tokenizer.json                           # HF tokenizer (full)
    └── config.json                              # Model dimensions

Target consumer: brevox-ios (on-device transcription)
"""

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()

MODEL_ID = "openai/whisper-large-v3-turbo"
N_MELS = 128
MEL_FRAMES = 3000  # 30s of audio at 16kHz → 3000 mel frames


def main():
    parser = argparse.ArgumentParser(
        description="Export Whisper large-v3-turbo encoder to CoreML + decoder assets"
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

    output_dir = Path(args.output_dir) / "Whisper_large_v3_turbo"
    output_dir.mkdir(parents=True, exist_ok=True)

    precision_label = "FP32" if args.no_half else "FP16"

    table = Table(title="Whisper large-v3-turbo Export Config")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Model", MODEL_ID)
    table.add_row("Mel Input", f"{N_MELS} × {MEL_FRAMES} (30s audio)")
    table.add_row("Encoder Precision", precision_label)
    table.add_row("Decoder", "Weights only (safetensors)")
    table.add_row("Output Dir", str(output_dir))
    console.print(table)

    # Step 1: Load model from HuggingFace
    console.print(f"\n[bold]Step 1/5: Loading {MODEL_ID}...[/bold]")
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32
    )
    model.eval()
    processor = WhisperProcessor.from_pretrained(MODEL_ID)

    cfg = model.config
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    console.print(f"  [green]Loaded ({n_params:.0f}M parameters)[/green]")
    console.print(
        f"  Encoder: {cfg.encoder_layers} layers, Decoder: {cfg.decoder_layers} layers"
    )

    # Step 2: Trace and convert encoder to CoreML
    console.print("[bold]Step 2/5: Exporting encoder to CoreML...[/bold]")

    encoder = model.get_encoder()
    encoder.eval()

    mel_input = torch.randn(1, N_MELS, MEL_FRAMES)

    # Wrap encoder to accept raw mel tensor (HF encoder expects input_features kwarg)
    class EncoderWrapper(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, mel_spectrogram):
            return self.encoder(mel_spectrogram).last_hidden_state

    wrapper = EncoderWrapper(encoder)
    wrapper.eval()

    with torch.no_grad():
        traced_encoder = torch.jit.trace(wrapper, mel_input)
        encoder_out = traced_encoder(mel_input)
    console.print(f"  Encoder output: [cyan]{list(encoder_out.shape)}[/cyan]")

    import coremltools as ct

    precision = ct.precision.FLOAT32 if args.no_half else ct.precision.FLOAT16

    ml_encoder = ct.convert(
        traced_encoder,
        inputs=[
            ct.TensorType(
                name="mel_spectrogram",
                shape=(1, N_MELS, MEL_FRAMES),
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
        f"Whisper large-v3-turbo encoder. "
        f"Input: mel spectrogram [{N_MELS}, {MEL_FRAMES}]. "
        f"Output: audio features [{cfg.max_source_positions}, {cfg.d_model}]."
    )

    encoder_path = output_dir / "WhisperEncoder_large_v3_turbo.mlpackage"
    if encoder_path.exists():
        import shutil

        shutil.rmtree(encoder_path)
    ml_encoder.save(str(encoder_path))
    console.print(f"  [green]Saved {encoder_path}[/green]")

    # Step 3: Export decoder weights
    console.print("[bold]Step 3/5: Exporting decoder weights...[/bold]")
    from safetensors.torch import save_file

    decoder_state = model.get_decoder().state_dict()
    # Include the output projection (lm_head) — needed for token prediction
    lm_head_state = {f"lm_head.{k}": v for k, v in model.proj_out.state_dict().items()}
    decoder_state.update(lm_head_state)

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

    tokenizer = processor.tokenizer
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save_pretrained(str(output_dir))
    # Also save as a single tokenizer.json for easy loading in Swift
    if (output_dir / "tokenizer.json").exists():
        console.print(f"  [green]Saved tokenizer to {tokenizer_path}[/green]")
    else:
        # Fallback: save vocab + special tokens manually
        vocab = tokenizer.get_vocab()
        special_tokens = {
            "bos_token_id": cfg.bos_token_id,
            "eos_token_id": cfg.eos_token_id,
            "decoder_start_token_id": cfg.decoder_start_token_id,
            "pad_token_id": cfg.pad_token_id,
        }
        tokenizer_data = {"vocab": vocab, "special_tokens": special_tokens}
        with open(tokenizer_path, "w") as f:
            json.dump(tokenizer_data, f, ensure_ascii=False)
        console.print(f"  [green]Saved {len(vocab)} vocab entries to {tokenizer_path}[/green]")

    vocab_size = len(tokenizer.get_vocab())
    console.print(f"  Vocab size: {vocab_size}")

    # Step 5: Export config
    console.print("[bold]Step 5/5: Exporting config...[/bold]")

    config = {
        "variant": "large-v3-turbo",
        "model_id": MODEL_ID,
        "n_mels": N_MELS,
        "n_audio_ctx": cfg.max_source_positions,
        "n_audio_state": cfg.d_model,
        "n_audio_head": cfg.encoder_attention_heads,
        "n_audio_layer": cfg.encoder_layers,
        "n_text_ctx": cfg.max_target_positions,
        "n_text_state": cfg.d_model,
        "n_text_head": cfg.decoder_attention_heads,
        "n_text_layer": cfg.decoder_layers,
        "n_vocab": cfg.vocab_size,
        "precision": precision_label,
        "mel_params": {
            "sample_rate": 16000,
            "n_fft": 400,
            "hop_length": 160,
            "n_mels": N_MELS,
            "chunk_length_s": 30,
        },
    }

    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    console.print(f"  [green]Saved {config_path}[/green]")

    # Summary
    console.print(f"\n[bold green]Export complete![/bold green]")

    info_table = Table(title="Whisper large-v3-turbo Export Summary")
    info_table.add_column("Property", style="cyan")
    info_table.add_column("Value", style="green")
    info_table.add_row("Model", MODEL_ID)
    info_table.add_row("Parameters", f"{n_params:.0f}M")
    info_table.add_row("Encoder", f"{cfg.encoder_layers} layers")
    info_table.add_row("Decoder", f"{cfg.decoder_layers} layers ({n_tensors} tensors)")
    info_table.add_row("Encoder Input", f"mel [{N_MELS}, {MEL_FRAMES}]")
    info_table.add_row(
        "Encoder Output",
        f"features [{cfg.max_source_positions}, {cfg.d_model}]",
    )
    info_table.add_row("Vocab", f"{vocab_size} tokens")
    info_table.add_row("Precision", precision_label)

    if encoder_path.is_dir():
        enc_size = sum(
            f.stat().st_size for f in encoder_path.rglob("*") if f.is_file()
        )
        info_table.add_row("Encoder Size", f"{enc_size / 1024 / 1024:.1f} MB")
    if weights_path.exists():
        dec_size = weights_path.stat().st_size
        info_table.add_row("Decoder Weights", f"{dec_size / 1024 / 1024:.1f} MB")

    console.print(info_table)

    console.print(f"\n[bold]Output directory:[/bold] {output_dir}/")
    console.print("  WhisperEncoder → CoreML (ANE inference)")
    console.print("  decoder_weights.safetensors → load in Swift (Accelerate)")
    console.print("  tokenizer.json → decode in Swift")
    console.print("  config.json → model dimensions for Swift decoder")

    console.print("\n[bold]Next step:[/bold] Copy to Brevox:")
    console.print(
        f"  cp -r {output_dir} ~/Developer/apps/brevox-ios/Resources/MLModels/"
    )

    console.print("\n[bold]Note:[/bold] Mel spectrogram: 128 mel bins (not 80!)")
    console.print("  Update Brevox mel computation: n_mels=128, 16kHz, hop=160, FFT=400")


if __name__ == "__main__":
    main()
