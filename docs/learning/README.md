# Learning — coreml-forge

Indice de guias tecnicas sobre exportacion de modelos ML a CoreML.

---

## Guias

| # | Archivo | Que cubre |
|---|---------|-----------|
| 1 | [yolo_coreml_export.md](yolo_coreml_export.md) | YOLO → CoreML: variantes (26s/11s/v8s), NMS, precision, input sizes, end-to-end vs legacy, performance en iPhone |
| 2 | [depth_anything_export.md](depth_anything_export.md) | Depth Anything V2 → CoreML: variantes (vits/vitb/vitl), tracing con JIT, conversion con coremltools, profundidad inversa |

**Orden de lectura**: 1 → 2

**Cuando leerlos**: Antes de modificar scripts de export o agregar nuevos modelos.

---

## Referencia cruzada con rayban-nav

| Guia | Consumidor en rayban-nav | Equivalente aria-demo |
|------|-------------------------|----------------------|
| yolo_coreml_export.md | `DetectorService.swift` — VNCoreMLRequest | `detector.py` — YOLO TensorRT |
| depth_anything_export.md | `DetectorService.swift` — pipeline paralelo | `detector.py` — Depth Anything TensorRT |
