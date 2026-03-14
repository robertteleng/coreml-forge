# CLAUDE.md — coreml-forge

## Project Identity
- **Tool**: coreml-forge — Export ML models to CoreML for iOS
- **Consumers**:
  - [rayban-nav](https://github.com/robertteleng/rayban-nav) — YOLO + Depth Anything V2 (visión)
  - [flow-coach-ios](../flow-coach-ios) — Face Landmarks (face/body)
  - [brevox-ios](../../apps/brevox-ios) — Whisper + Summarizer (audio/lenguaje)
- **Runtime**: Python 3.10+, managed with `uv`

## Quick Commands
```bash
# Install dependencies
uv sync

# Export YOLO (default: yolo26s, 640x640, FP16)
uv run python scripts/export_yolo.py

# Export Depth Anything V2 (default: vits, 518x518, FP16)
uv run python scripts/export_depth.py

# Export Face Landmarks (468-point, 192x192, FP16, ANE)
uv run python scripts/export_face_landmarks.py

# Export Whisper small (FP16, encoder-only CoreML + decoder weights)
uv run python scripts/export_whisper.py

# Export Whisper large-v3-turbo (FP16, 128 mel bins, 4 decoder layers)
uv run python scripts/export_whisper_turbo.py
```

## Structure
```
coreml-forge/
├── scripts/
│   ├── export_yolo.py             # YOLO → CoreML (default: yolo26s)
│   ├── export_depth.py            # Depth Anything V2 → CoreML (default: vits)
│   ├── export_face_landmarks.py   # Face Landmarks → CoreML (468-point, ANE)
│   ├── export_whisper.py          # Whisper → CoreML encoder + decoder weights
│   ├── export_whisper_turbo.py   # Whisper large-v3-turbo → CoreML (HF transformers)
│   └── export_summarizer.py      # Qwen3.5-4B → CoreML via ANEMLL (macOS only)
├── exports/                        # Output models (gitignored)
├── .models/                        # Auto-cloned repos + downloaded weights (gitignored)
├── docs/
│   ├── project/
│   │   ├── IMPLEMENTATION_PLAN.md  # Plan por consumer (rayban-nav, flow-coach, brevox)
│   │   ├── PAIR_WORKFLOW.md
│   │   ├── DEVELOPER_DIARY.md
│   │   ├── CHANGELOG.md
│   │   └── WORKFLOW.md
│   └── learning/
│       ├── yolo_coreml_export.md
│       ├── depth_anything_export.md
│       └── whisper_coreml_export.md
├── pyproject.toml                  # uv project config + dependencies
└── CLAUDE.md                       # ← You are here
```

## Models

| Model | Script | Default Config | Size | Consumer |
|-------|--------|---------------|------|----------|
| YOLO26s | `export_yolo.py` | 640x640 FP16, NMS-free | ~18 MB | rayban-nav |
| Depth Anything V2 vits | `export_depth.py` | 518x518 FP16, iOS 18 | ~47 MB | rayban-nav |
| Face Landmarks | `export_face_landmarks.py` | 192x192 FP16, CPU+ANE, 468 pts | ~1.2 MB | flow-coach-ios |
| Whisper small (encoder) | `export_whisper.py` | 80×3000 mel FP16, iOS 18 | ~230 MB | brevox-ios |
| Whisper large-v3-turbo | `export_whisper_turbo.py` | 128×3000 mel FP16, iOS 18 | ~1.2 GB | brevox-ios |
| Qwen3.5-4B (summarizer) | `export_summarizer.py` | ctx2048, LUT4+LUT6, ANEMLL, macOS only | ~2.5 GB | brevox-ios |

## Rules
- Use `uv` for everything (not pip, not conda)
- Exports go to `exports/` (gitignored — models are large binaries)
- Model repo clones go to `.models/` (gitignored)
- Scripts should be standalone with `--help` and sensible defaults
- Default to FP16 precision and iOS 18 deployment target
- Print model info (size, input shape, precision) after export
- Commits: `type(scope): description` — no Co-Authored-By trailers
- See `docs/project/WORKFLOW.md` for full conventions
