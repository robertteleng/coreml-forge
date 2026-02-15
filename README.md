# coreml-forge

Export ML models to CoreML (.mlpackage) for iOS apps.

## Setup

```bash
uv sync
```

## Usage

### YOLO (Object Detection)

```bash
uv run python scripts/export_yolo.py                    # yolo11s, 640x640, FP16
uv run python scripts/export_yolo.py --model yolov8s    # Different model
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

| Model | Script | Default | Size |
|-------|--------|---------|------|
| YOLO11s | `export_yolo.py` | 640x640 FP16 + NMS | ~20MB |
| Depth Anything V2 vits | `export_depth.py` | 518x518 FP16 | ~50MB |
