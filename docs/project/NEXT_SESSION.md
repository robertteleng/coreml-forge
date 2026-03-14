# NEXT SESSION

**Updated:** 2026-03-15

## Completed

### ~~1. Export Whisper large-v3-turbo to CoreML~~ ✓
- Exported via `scripts/export_whisper_turbo.py` (uses HF `transformers`)
- Encoder: 1215 MB `.mlpackage` (32 layers, FP16, ANE)
- Decoder: 455 MB safetensors (4 layers + lm_head, 101 tensors)
- **Importante para Brevox**: mel spectrogram ahora usa **128 bins** (no 80)
  - Actualizar `n_mels=128` en la computación de mel en Swift

## Pending Tasks

### 1. Copy Whisper large-v3-turbo to Brevox
- Copy `exports/Whisper_large_v3_turbo/` to brevox-ios bundle
- Update mel computation: `n_mels=128` (was 80)
- Update decoder Swift code for new dimensions: `d_model=1280`, `n_text_layer=4`

### 2. Summarizer Qwen3.5-4B Export (in progress)
- ANEMLL export running on Mac (16GB) with workaround:
  - Steps 1-3: `--batch 64 --chunk 4`
  - Step 4 (prefill): `--batch 8 --chunk 4` (fixes OOM on 16GB)
  - Steps 5-8: `--batch 64`
- If successful, copy `.mlpackage` files to Brevox bundle

### 3. Document ANEMLL 16GB Workaround
- The prefill step allocates KV cache with `batch_size` in the state shape
- `--batch 8` for Step 4 reduces KV state from ~16 GB to ~2 GB
- `--chunk 4` splits FFN layers into 4 parts (~8 GB each vs 32 GB)
- Clean `/var/folders/.../T/tmp*.mlpackage` after failed runs (temporals accumulate)

## Hardware Notes
- Mac M1 Pro 16GB: needs `--chunk 4` + `--batch 8` for Step 4
- NUC Linux 32GB: can run coremltools puro (no ANEMLL), then compile on Mac via Xcode
- 5060 Ti 16GB: useful for inference testing, not for CoreML export
