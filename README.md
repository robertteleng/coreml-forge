# coreml-forge

Export ML models to CoreML (.mlpackage) for iOS apps.

**Consumers:**
- [rayban-nav](https://github.com/robertteleng/rayban-nav) — YOLO + Depth Anything V2 (visual assistance)
- [flow-coach-ios](../flow-coach-ios) — Face Landmarks (dance coaching)
- [brevox-ios](../../apps/brevox-ios) — Whisper + Summarizer (voice recorder)

## Setup

```bash
uv sync
```

## Usage

### YOLO (Object Detection)

```bash
uv run python scripts/export_yolo.py                    # yolo26s, 640x640, FP16
uv run python scripts/export_yolo.py --model yolo26n    # Nano (faster)
```

### Depth Anything V2 (Depth Estimation)

```bash
uv run python scripts/export_depth.py                   # vits, 518x518, FP16
uv run python scripts/export_depth.py --variant vitb    # Larger model
```

### Face Landmarks (468-point, ANE)

```bash
uv run python scripts/export_face_landmarks.py          # FP16, CPU+ANE
uv run python scripts/export_face_landmarks.py --int8   # INT8 (2x faster on ANE)
```

### Whisper (Transcription)

```bash
uv run python scripts/export_whisper.py                 # small, FP16 (encoder CoreML + decoder weights)
uv run python scripts/export_whisper.py --variant tiny   # Tiny (lighter)
```

### Quantization (any model)

```bash
uv run python scripts/quantize_model.py exports/model.mlpackage              # INT8 weights
uv run python scripts/quantize_model.py exports/model.mlpackage --mode w8a8  # W8A8
```

## Output

Exported models go to `exports/` (tracked with Git LFS).

## Supported Models

| Model | Script | Default | Size | Consumer |
|-------|--------|---------|------|----------|
| **YOLO26s** | `export_yolo.py` | 640x640 FP16, NMS-free | ~18 MB | rayban-nav |
| **Depth Anything V2 vits** | `export_depth.py` | 518x518 FP16 | ~47 MB | rayban-nav |
| **Face Landmarks** | `export_face_landmarks.py` | 192x192 FP16, ANE | ~1.2 MB | flow-coach-ios |
| **Whisper (encoder)** | `export_whisper.py` | 80×3000 mel FP16 | ~230 MB (small) | brevox-ios |

## Documentation

| Document | Purpose |
|----------|---------|
| [CLAUDE.md](CLAUDE.md) | Rules, quick commands, project structure |
| [IMPLEMENTATION_PLAN](docs/project/IMPLEMENTATION_PLAN.md) | Roadmap by consumer |
| [WORKFLOW](docs/project/WORKFLOW.md) | Git conventions, script structure |
| [PAIR_WORKFLOW](docs/project/PAIR_WORKFLOW.md) | Collaboration workflow |
| [Learning guides](docs/learning/) | Technical guides on CoreML export |
