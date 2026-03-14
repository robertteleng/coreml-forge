# NEXT SESSION

**Updated:** 2026-03-15 00:20 UTC

## Pending Tasks

### 1. Export Whisper large-v3-turbo to CoreML
- Model: `openai/whisper-large-v3-turbo`
- Currently using Whisper small in Brevox — upgrade to large-v3-turbo for better accuracy
- large-v3-turbo has only 4 decoder layers (vs 32 in large-v3), so it's fast despite being "large"
- Export encoder as `.mlpackage` for ANE, decoder weights as `.safetensors`
- Same pipeline as Whisper small export — just different model
- Target: replace `Brevox/Resources/Models/Whisper_small/` with `Whisper_large_v3_turbo/`

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
