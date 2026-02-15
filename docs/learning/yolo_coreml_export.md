# YOLO → CoreML Export

Guia tecnica sobre exportar modelos YOLO a CoreML para iOS.

---

## Modelos Soportados

| Modelo | Params | GFLOPs | Output Shape | NMS | Notas |
|--------|--------|--------|-------------|-----|-------|
| **YOLO26s** (default) | 10.0M | 22.8 | (1, 300, 6) | End-to-end | Mas reciente, NMS-free, optimizado para edge |
| YOLO26n | ~3M | ~7 | (1, 300, 6) | End-to-end | Nano, mas rapido, menos preciso |
| YOLO11s | 9.4M | 21.5 | (1, 84, 8400) | Incluido via flag | Legacy, requiere `nms=True` |
| YOLOv8s | ~11M | ~28 | (1, 84, 8400) | Incluido via flag | Legacy |

---

## YOLO26 vs YOLO11

YOLO26 (septiembre 2025) es un cambio significativo:

| Aspecto | YOLO26 | YOLO11 |
|---------|--------|--------|
| NMS | End-to-end (built-in) | Requiere `nms=True` en export |
| DFL | Eliminado | Incluido |
| Output | `(1, 300, 6)` — 300 detecciones, 6 valores cada una | `(1, 84, 8400)` — raw predictions |
| CPU inference | 43% mas rapido | Baseline |
| Edge optimization | Diseniado para edge | General purpose |

**Para rayban-nav usamos YOLO26s** — es el default y el recomendado.

---

## Export Command

```bash
# Default: yolo26s, 640x640, FP16
uv run python scripts/export_yolo.py

# Opciones
uv run python scripts/export_yolo.py --model yolo26n    # Nano (mas rapido)
uv run python scripts/export_yolo.py --model yolo11s     # Legacy
uv run python scripts/export_yolo.py --imgsz 320         # Input mas pequenio
uv run python scripts/export_yolo.py --no-half           # FP32
```

---

## Como Funciona el Export

```python
from ultralytics import YOLO

# 1. Cargar modelo (descarga weights automaticamente)
model = YOLO("yolo26s.pt")

# 2. Exportar a CoreML
model.export(
    format="coreml",
    nms=True,        # Ignorado por YOLO26 (end-to-end)
    imgsz=640,
    half=True,       # FP16
)
```

Ultralytics maneja todo internamente:
1. Trace del modelo PyTorch
2. Conversion via coremltools
3. Optimizacion de pipeline
4. Output: `.mlpackage`

---

## Precision: FP16 vs FP32

| Aspecto | FP16 (default) | FP32 |
|---------|----------------|------|
| Tamano | ~18 MB | ~36 MB |
| Neural Engine | Optimizado | Funciona pero mas lento |
| Precision | Suficiente para deteccion | Marginal mejora |
| Recomendado | Si | Solo para debugging |

Apple Neural Engine esta optimizado para FP16. No hay razon practica para usar FP32 en produccion.

---

## Input Size: 640 vs 320

| Aspecto | 640x640 (default) | 320x320 |
|---------|-------------------|---------|
| Precision | Mayor | Menor (objetos pequenios se pierden) |
| FPS | ~30-40 FPS (iPhone 15 Pro) | ~50-60 FPS |
| Tamano modelo | Igual | Igual |
| Recomendado | Si | Solo si latencia es critica |

El streaming de DAT SDK es ~504x896 en `.low`. 640x640 es un buen match.

---

## Consumo en rayban-nav

En `DetectorService.swift`:

```swift
// Carga del modelo
let mlModel = try MLModel(contentsOf: yolo26sURL)
let visionModel = try VNCoreMLModel(for: mlModel)

// Inferencia
let request = VNCoreMLRequest(model: visionModel) { request, error in
    guard let results = request.results as? [VNRecognizedObjectObservation] else { return }
    // YOLO26 output: 300 detecciones max, ya filtradas por NMS
    for observation in results {
        let label = observation.labels.first?.identifier ?? ""
        let confidence = observation.labels.first?.confidence ?? 0
        let bbox = observation.boundingBox // CGRect normalizado (0-1)
    }
}
```

**Importante**: Vision framework usa origen bottom-left. rayban-nav convierte a top-left en `Detection.swift`.

---

## Comparacion con aria-demo

| Aspecto | aria-demo (Python) | coreml-forge → rayban-nav |
|---------|-------------------|--------------------------|
| Modelo | YOLOv26s TensorRT | YOLO26s CoreML |
| Runtime | TensorRT (CUDA) | CoreML (Neural Engine) |
| Precision | FP16 | FP16 |
| Input | 640x640 | 640x640 |
| NMS | Built-in (end-to-end) | Built-in (end-to-end) |
| Confidence threshold | 0.4 | 0.4 (en DetectorService) |
| FPS | ~60 FPS (RTX 3060) | ~30-40 FPS (iPhone 15 Pro) |
