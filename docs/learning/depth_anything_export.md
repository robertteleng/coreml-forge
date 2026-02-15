# Depth Anything V2 → CoreML Export

Guia tecnica sobre exportar Depth Anything V2 a CoreML para estimacion de profundidad monocular en iOS.

---

## Por Que Depth Estimation

Las Ray-Ban Meta no tienen sensor de profundidad (LiDAR/ToF). Para estimar distancia a objetos detectados, usamos **estimacion de profundidad monocular** — un modelo ML que predice profundidad relativa desde una sola imagen RGB.

---

## Variantes del Modelo

| Variante | Params | Encoder | Tamano Export | FPS (iPhone 15 Pro) | Recomendado |
|----------|--------|---------|--------------|-------------------|-------------|
| **vits** (default) | 24.8M | DINOv2-S | ~47 MB | ~15-25 FPS | Si — balance velocidad/precision |
| vitb | 97.5M | DINOv2-B | ~180 MB | ~8-12 FPS | Para mayor precision |
| vitl | 335.3M | DINOv2-L | ~600 MB | ~3-5 FPS | No recomendado para mobile |

**Para rayban-nav usamos vits** — suficiente precision para categorizar distancias (very_close/close/medium/far).

---

## Export Command

```bash
# Default: vits, 518x518, FP16
uv run python scripts/export_depth.py

# Opciones
uv run python scripts/export_depth.py --variant vitb    # Base (mas grande)
uv run python scripts/export_depth.py --imgsz 256       # Mas rapido
uv run python scripts/export_depth.py --no-half         # FP32
```

---

## Como Funciona el Export

A diferencia de YOLO (que Ultralytics exporta directamente), Depth Anything V2 requiere pasos manuales:

```python
import torch
import coremltools as ct
from depth_anything_v2.dpt import DepthAnythingV2

# 1. Cargar modelo PyTorch
model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
model.load_state_dict(torch.load("depth_anything_v2_vits.pth", map_location="cpu"))
model.eval()

# 2. Trace con JIT (input fijo)
dummy_input = torch.randn(1, 3, 518, 518)
traced = torch.jit.trace(model, dummy_input)

# 3. Convertir a CoreML
mlmodel = ct.convert(
    traced,
    inputs=[ct.ImageType(name="image", shape=(1, 3, 518, 518), scale=1/255.0)],
    compute_precision=ct.precision.FLOAT16,
    minimum_deployment_target=ct.target.iOS18,
)

# 4. Guardar
mlmodel.save("DepthAnythingV2_vits.mlpackage")
```

### Pasos automaticos del script
1. **Clone** del repo Depth Anything V2 (si no existe)
2. **Download** de weights desde HuggingFace
3. **Load** del modelo PyTorch
4. **Trace** con `torch.jit.trace`
5. **Convert** con `coremltools`
6. **Save** como `.mlpackage`

---

## Profundidad Inversa

**Concepto critico**: El output del modelo es **profundidad inversa** — valores mas altos significan **mas cerca**.

```
Output: depth_map[y][x] = valor float

Alto valor (>0.7)  → Objeto MUY cerca
Medio valor (0.3-0.7) → Objeto a distancia media
Bajo valor (<0.3)  → Objeto lejos
```

En rayban-nav, `DistanceCategory` mapea estos valores:

| Categoria | Threshold | Significado |
|-----------|----------|-------------|
| `veryClose` | > 0.7 | Peligro inmediato |
| `close` | > 0.5 | Atencion requerida |
| `medium` | > 0.3 | Awareness |
| `far` | <= 0.3 | No alertar |

**Nota**: Estos thresholds vienen de aria-demo y pueden necesitar calibracion para la resolucion del DAT SDK (~504x896 vs 1408x1408 de Aria).

---

## Input Size: 518 vs 256

| Aspecto | 518x518 (default) | 256x256 |
|---------|-------------------|---------|
| Precision depth | Mayor | Menor |
| FPS | ~15-25 | ~30-40 |
| Tamano modelo | Igual | Igual |
| Recomendado | Si | Si latencia es critica |

518 es el tamano nativo del modelo (14 patches x 37 = 518). Otros tamanos funcionan pero deben ser multiplo de 14.

---

## Consumo en rayban-nav

En `DetectorService.swift`, pipeline paralelo con YOLO:

```swift
// Inferencia paralela
async let detections = detectObjects(in: pixelBuffer)
async let depthMap = estimateDepth(in: pixelBuffer)

let (objects, depth) = await (detections, depthMap)

// Enriquecer detecciones con profundidad
for var detection in objects {
    let depthValue = extractDepth(from: depth, at: detection.bbox)
    detection.depth = depthValue
    detection.distance = DistanceCategory.from(depth: depthValue)
}
```

---

## Comparacion con aria-demo

| Aspecto | aria-demo (Python) | coreml-forge → rayban-nav |
|---------|-------------------|--------------------------|
| Modelo | Depth Anything V2 vits | Depth Anything V2 vits |
| Runtime | TensorRT (CUDA) | CoreML (Neural Engine) |
| Precision | FP16 | FP16 |
| Input | 518x518 | 518x518 |
| Output | Depth map (inverse) | Depth map (inverse) |
| Normalizacion | min-max per frame | min-max per frame |
| FPS | ~40 FPS (RTX 3060) | ~15-25 FPS (iPhone 15 Pro) |
| Threshold confidence | 0.4 | 0.4 |
| Distance categories | very_close/close/medium/far | very_close/close/medium/far |
