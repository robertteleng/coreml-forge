# Plan de Implementacion — coreml-forge

Modelos ML a exportar para [rayban-nav](https://github.com/robertteleng/rayban-nav) (iOS, CoreML/Vision).

---

## Fase 1: Modelos Core

**Objetivo**: Exportar los dos modelos principales que necesita rayban-nav para deteccion + profundidad.

**Equivalente aria-demo**: `detector.py` usa YOLO TensorRT + Depth Anything V2 TensorRT.

### Hito 1.1 — YOLO Object Detection
- [x] Script `export_yolo.py` con CLI completo (`--model`, `--imgsz`, `--no-nms`, `--no-half`)
- [x] Default: YOLO26s, 640x640, FP16
- [x] Export verificado: 18.4 MB, 80 clases, end-to-end (NMS-free)
- [x] Copiado a rayban-nav `Resources/MLModels/`

### Hito 1.2 — Depth Estimation
- [x] Script `export_depth.py` con CLI completo (`--variant`, `--imgsz`, `--no-half`)
- [x] Default: vits (Small), 518x518, FP16
- [x] Auto-clone de repo Depth Anything V2 + descarga de weights
- [x] Export verificado: 47.3 MB, depth map output
- [x] Copiado a rayban-nav `Resources/MLModels/`

---

## Fase 2: Optimizacion y Variantes (Pendiente)

**Objetivo**: Explorar modelos mas rapidos/ligeros si el pipeline de rayban-nav lo requiere.

### Hito 2.1 — Variantes de tamano
- [ ] Exportar YOLO26n (Nano) — mas rapido, menos preciso
- [ ] Exportar Depth Anything V2 con imgsz 256 — mas rapido
- [ ] Benchmark: comparar FPS y precision vs variantes Small

### Hito 2.2 — Quantizacion
- [ ] Evaluar INT8 quantization para ambos modelos
- [ ] Medir impacto en precision vs tamanio/velocidad
- [ ] Documentar resultados en learning/

### Hito 2.3 — Nuevos modelos
- [ ] Evaluar modelos de segmentacion si rayban-nav lo necesita
- [ ] Evaluar modelos de pose estimation (YOLO26-pose)

---

## Notas

- **Consumer unico**: rayban-nav es el unico consumidor. Los exports se optimizan para su pipeline especifico.
- **Apple Silicon**: Los exports se generan en Mac (CPU). CoreML compila para Neural Engine en el device.
- **Tamano importa**: Los modelos van bundled en la app iOS. Menos MB = app mas ligera.
- **FPS target**: rayban-nav necesita ~15-30 FPS para deteccion en tiempo real. Los modelos Small son suficientes.
