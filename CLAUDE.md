# CLAUDE.md — coreml-forge

## Project Identity
- **Tool**: coreml-forge — Export ML models to CoreML for iOS
- **Primary consumer**: [rayban-nav](https://github.com/robertteleng/rayban-nav) (YOLO + Depth Anything V2)
- **Runtime**: Python 3.10+, managed with `uv`

## Quick Commands
```bash
# Install dependencies
uv sync

# Export YOLO
uv run python scripts/export_yolo.py

# Export Depth Anything V2
uv run python scripts/export_depth.py

# Copy to rayban-nav
cp -r exports/*.mlpackage ~/Developer/extreme/rayban-nav/RayBanNav/Resources/MLModels/
```

## Structure
```
coreml-forge/
├── scripts/
│   ├── export_yolo.py        # YOLO → CoreML
│   └── export_depth.py       # Depth Anything V2 → CoreML
├── exports/                   # Output .mlpackage (gitignored)
├── docs/
│   └── DOCUMENTATION_GUIDE.md
├── pyproject.toml             # uv project config + dependencies
└── CLAUDE.md                  # ← You are here
```

## Rules
- Use `uv` for everything (not pip, not conda)
- Exports go to `exports/` (gitignored — models are large binaries)
- Scripts should be standalone with `--help` and sensible defaults
- Default to FP16 precision and iOS 18 deployment target
- Print model info (size, input shape, precision) after export
