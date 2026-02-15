# coreml-forge

Export ML models to CoreML (.mlpackage) for iOS apps.

Consumer: [rayban-nav](https://github.com/robertteleng/rayban-nav) — visual assistance for visually impaired users using Meta Ray-Ban smart glasses.

## Setup

```bash
uv sync
```

## Usage

### YOLO (Object Detection)

```bash
uv run python scripts/export_yolo.py                    # yolo26s, 640x640, FP16
uv run python scripts/export_yolo.py --model yolo26n    # Nano (faster)
uv run python scripts/export_yolo.py --model yolo11s    # Legacy model
uv run python scripts/export_yolo.py --imgsz 320        # Faster inference
```

### Depth Anything V2 (Depth Estimation)

```bash
uv run python scripts/export_depth.py                   # vits, 518x518, FP16
uv run python scripts/export_depth.py --variant vitb    # Larger model
uv run python scripts/export_depth.py --imgsz 256       # Faster inference
```

## Output

Exported models go to `exports/`. Copy to your Xcode project:

```bash
cp -r exports/*.mlpackage ~/Developer/extreme/rayban-nav/RayBanNav/Resources/MLModels/
```

## Supported Models

| Model | Script | Default | Size | Notes |
|-------|--------|---------|------|-------|
| **YOLO26s** | `export_yolo.py` | 640x640 FP16 | ~18 MB | End-to-end (NMS-free), optimized for edge |
| Depth Anything V2 vits | `export_depth.py` | 518x518 FP16 | ~47 MB | Monocular depth estimation |

## Documentation

| Document | Purpose |
|----------|---------|
| [CLAUDE.md](CLAUDE.md) | Rules, quick commands, project structure |
| [DOCUMENTATION_GUIDE](docs/project/DOCUMENTATION_GUIDE.md) | Map of all documents |
| [IMPLEMENTATION_PLAN](docs/project/IMPLEMENTATION_PLAN.md) | Roadmap of models to export |
| [WORKFLOW](docs/project/WORKFLOW.md) | Git conventions, script structure |
| [CHANGELOG](docs/project/CHANGELOG.md) | Development history |
| [Learning guides](docs/learning/README.md) | Technical guides on CoreML export |

## Origin

Port of the ML export pipeline from [aria-demo](https://github.com/robertteleng/aria-demo) (Python/CUDA/TensorRT) to CoreML for iOS deployment.

| aria-demo | coreml-forge |
|-----------|-------------|
| YOLO TensorRT (.engine) | YOLO CoreML (.mlpackage) |
| Depth Anything V2 TensorRT | Depth Anything V2 CoreML |
| GPU: NVIDIA CUDA | GPU: Apple Neural Engine |
| Runtime: Python | Runtime: Swift (via rayban-nav) |
