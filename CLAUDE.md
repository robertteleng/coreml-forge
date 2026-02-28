# CLAUDE.md — coreml-forge

## Project Identity
- **Tool**: coreml-forge — Export ML models to CoreML for iOS
- **Primary consumers**: [rayban-nav](https://github.com/robertteleng/rayban-nav) (YOLO + Depth Anything V2), [flow-coach-ios](../flow-coach-ios) (Face Landmarks)
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

# Copy to rayban-nav
cp -r exports/*.mlpackage ~/Developer/extreme/rayban-nav/RayBanNav/Resources/MLModels/
```

## Structure
```
coreml-forge/
├── scripts/
│   ├── export_yolo.py        # YOLO → CoreML (default: yolo26s)
│   ├── export_depth.py       # Depth Anything V2 → CoreML (default: vits)
│   └── export_face_landmarks.py  # Face Landmarks → CoreML (468-point, ANE)
├── exports/                   # Output .mlpackage (gitignored)
├── docs/
│   ├── project/
│   │   ├── DOCUMENTATION_GUIDE.md
│   │   ├── CHANGELOG.md
│   │   ├── IMPLEMENTATION_PLAN.md
│   │   └── WORKFLOW.md
│   └── learning/
│       ├── README.md
│       ├── yolo_coreml_export.md
│       └── depth_anything_export.md
├── pyproject.toml             # uv project config + dependencies
└── CLAUDE.md                  # ← You are here
```

## Models

| Model | Script | Default Config | Size |
|-------|--------|---------------|------|
| YOLO26s | `export_yolo.py` | 640x640 FP16, end-to-end (NMS-free) | ~18 MB |
| Depth Anything V2 vits | `export_depth.py` | 518x518 FP16, iOS 18 | ~47 MB |
| Face Landmarks | `export_face_landmarks.py` | 192x192 FP16, CPU+ANE, 468 points | ~1.2 MB |

## Rules
- Use `uv` for everything (not pip, not conda)
- Exports go to `exports/` (gitignored — models are large binaries)
- Scripts should be standalone with `--help` and sensible defaults
- Default to FP16 precision and iOS 18 deployment target
- Print model info (size, input shape, precision) after export
- Commits: `type(scope): description` — no Co-Authored-By trailers
- See `docs/project/WORKFLOW.md` for full conventions
