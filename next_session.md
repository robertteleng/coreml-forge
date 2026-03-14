# Next Session — Whisper + Summarizer → CoreML for Brevox

**Consumer**: [brevox-ios](../../apps/brevox-ios) — Voice recorder app with local transcription + summarization
**Priority**: Whisper first (core feature), then summarizer
**Goal**: Export Whisper tiny/base and a small language model to CoreML for on-device inference

---

## Context

Brevox is a local-first voice recording app (iOS 18+, iPhone 15 Pro minimum).
Currently the transcription and summarization engines use fallback heuristics.
We need real Core ML models to ship with the app.

| Model | Purpose | Current State | Target |
|-------|---------|---------------|--------|
| **Whisper** | Audio → transcript | `CoreMLWhisperTranscriber` with debug fallback | Real Core ML inference |
| **Summarizer** | Transcript → summary/highlights/actions | `HeuristicSummarizationEngine` fallback | Real Core ML inference |

---

## 1. Whisper Export (Priority 1)

### Model Selection

| Variant | Parameters | Size (FP16) | Quality | Speed (A17) |
|---------|-----------|-------------|---------|-------------|
| **whisper-tiny** | 39M | ~75 MB | Good for short recordings | ~1x real-time |
| **whisper-base** | 74M | ~140 MB | Better accuracy | ~0.5x real-time |
| whisper-small | 244M | ~460 MB | High quality | Slow on-device |

**Recommendation**: Start with `whisper-tiny`, benchmark, then try `whisper-base` if quality insufficient.

### Architecture

Whisper has 3 components:
1. **Audio preprocessor** — mel spectrogram (80 bins, 16kHz, 30s window)
2. **Encoder** — audio features → latent representation
3. **Decoder** — autoregressive token generation

### Export Strategy

Two approaches:

**A) Export encoder + decoder separately (recommended)**
```python
# Encoder: audio mel → features
encoder_model = whisper.model.encoder
traced_encoder = torch.jit.trace(encoder_model, mel_input)
ct.convert(traced_encoder, ...)  → WhisperEncoder.mlpackage

# Decoder: features + tokens → next token
decoder_model = whisper.model.decoder
traced_decoder = torch.jit.trace(decoder_model, (features, tokens))
ct.convert(traced_decoder, ...)  → WhisperDecoder.mlpackage
```

**B) Use `whisper-coreml` community export (faster path)**
- Check `huggingface.co/coreml-community/whisper-tiny` for pre-converted models
- Validate output format matches our `TranscriptSegment(start, end, text)`

### Script to Create

```
scripts/export_whisper.py
```

### Interface

```bash
uv run python scripts/export_whisper.py                          # Default: tiny, FP16
uv run python scripts/export_whisper.py --variant base           # Base model
uv run python scripts/export_whisper.py --no-half                # FP32
```

### Output

```
exports/WhisperEncoder.mlpackage
exports/WhisperDecoder.mlpackage
```

### Key Considerations

1. **Mel spectrogram** — can be computed in Swift with `Accelerate.framework` (FFT + mel filterbank) or exported as a third Core ML model
2. **Tokenizer** — need the Whisper tokenizer (BPE). Options: port to Swift, or bundle tokenizer vocab + decode in Swift
3. **Language detection** — Whisper auto-detects language. Keep this behavior for multi-language support
4. **30-second chunks** — Whisper processes 30s windows. For longer recordings, chunk audio and concatenate segments
5. **Timestamps** — Whisper can output word-level timestamps, map these to `TranscriptSegment.start/end`

---

## 2. Summarizer Export (Priority 2)

### Model Selection

| Model | Parameters | Size (FP16) | Quality | Speed |
|-------|-----------|-------------|---------|-------|
| **Phi-3.5-mini-instruct** | 3.8B | ~7 GB | Excellent | Slow on-device |
| **Qwen2.5-0.5B-Instruct** | 0.5B | ~1 GB | Good | Reasonable |
| **SmolLM2-360M-Instruct** | 360M | ~700 MB | Decent | Fast |
| **TinyLlama-1.1B** | 1.1B | ~2 GB | Good | Moderate |

**Recommendation**: Start with `SmolLM2-360M-Instruct` or `Qwen2.5-0.5B-Instruct`. Small enough for on-device, capable enough for structured JSON output (summary + highlights + actions).

### Export Strategy

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import coremltools as ct

model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")

# Trace and convert
traced = torch.jit.trace(model, dummy_input)
mlmodel = ct.convert(traced, minimum_deployment_target=ct.target.iOS18, ...)
mlmodel.save("exports/SmolLM2.mlpackage")
```

### Script to Create

```
scripts/export_summarizer.py
```

### Interface

```bash
uv run python scripts/export_summarizer.py                              # Default: SmolLM2-360M
uv run python scripts/export_summarizer.py --variant qwen2.5-0.5b      # Qwen variant
uv run python scripts/export_summarizer.py --no-half                    # FP32
```

### Output

```
exports/SmolLM2.mlpackage    # or Qwen2.mlpackage
```

### Key Considerations

1. **Tokenizer** — bundle tokenizer vocab JSON in iOS app, decode in Swift
2. **Prompt format** — match the existing prompt in `SummarizationEngine.swift` (JSON output: summary, highlights, actionItems)
3. **Context window** — 360M models handle ~2K tokens well. Truncate long transcripts to fit
4. **Quantization** — INT8 or INT4 via `coremltools.optimize` to reduce model size and improve ANE speed
5. **On-demand download** — if model is >100MB, consider Apple's on-demand resources or background asset download

---

## Integration in Brevox

### Whisper
1. Add `WhisperEncoder.mlpackage` + `WhisperDecoder.mlpackage` to Xcode bundle
2. Implement mel spectrogram in `CoreMLWhisperTranscriber` using `Accelerate`
3. Run encoder → decoder loop with greedy/beam search
4. Map tokens → text segments with timestamps
5. Remove `DebugTranscriptionEngine` fallback (keep as test-only)

### Summarizer
1. Add `SmolLM2.mlpackage` to Xcode bundle (or on-demand resource)
2. Create `CoreMLSummarizationEngine: SummarizationEngine`
3. Tokenize input, run inference, decode JSON output
4. Keep `HeuristicSummarizationEngine` as fallback if model unavailable

### Files to Modify in Brevox
- `Brevox/Services/Processing/CoreMLWhisperTranscriber.swift` — real pipeline
- `Brevox/Services/Processing/SummarizationEngine.swift` — new Core ML engine
- `project.yml` — add model resources to bundle
- `CLAUDE.md` — update model references

---

## Success Criteria

- [ ] Whisper tiny exported to Core ML (encoder + decoder)
- [ ] Whisper transcribes a 30s audio clip correctly on device
- [ ] Whisper auto-detects language (Spanish + English minimum)
- [ ] Summarizer exported to Core ML
- [ ] Summarizer produces valid JSON (summary, highlights, actionItems) from transcript
- [ ] Both models run on iPhone 15 Pro simulator
- [ ] Total model bundle < 500 MB (or split with on-demand resources)

---

## Previous Tasks (Completed)

### Face Landmarks (flow-coach-ios)
- [x] PyTorch FaceMesh model loads and traces cleanly
- [x] CoreML export succeeds with FP16 and CPU_AND_NE (1.2 MB)
- [ ] Output landmarks match MediaPipe 468-point indices
- [ ] Model runs on ANE (verify with Instruments)

### Later: Hand + Pose (flow-coach-ios)
Same PyTorch→CoreML pipeline. Lower priority than Brevox models.
