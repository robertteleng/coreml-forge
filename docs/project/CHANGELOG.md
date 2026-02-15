# Changelog — coreml-forge

Registro cronologico de desarrollo, decisiones tecnicas, y cambios significativos.

---

## 2026-02-15 — Setup Inicial + Primeros Exports

### Creado

**Proyecto**:
- `pyproject.toml` — Config uv con dependencies: ultralytics, coremltools, torch, torchvision, huggingface-hub, rich
- `.gitignore` — Ignora exports/*.mlpackage, *.pt, *.pth, Depth-Anything-V2/
- `.python-version` — Python 3.10.2
- `exports/.gitkeep` — Directorio de output (modelos gitignored)

**Scripts**:
- `scripts/export_yolo.py` — YOLO → CoreML. Default: yolo26s, 640x640, FP16. Soporta cualquier modelo YOLO via `--model`. Incluye NMS handling (YOLO26 es end-to-end, no necesita `--nms`).
- `scripts/export_depth.py` — Depth Anything V2 → CoreML. Default: vits, 518x518, FP16. Clona repo + descarga weights de HuggingFace automaticamente.

**Documentacion**:
- `CLAUDE.md` — Reglas del proyecto, quick commands, estructura
- `README.md` — Overview, setup, uso, tabla de modelos
- `docs/project/DOCUMENTATION_GUIDE.md` — Mapa de documentos
- `docs/project/CHANGELOG.md` — Este archivo
- `docs/project/IMPLEMENTATION_PLAN.md` — Roadmap de modelos
- `docs/project/WORKFLOW.md` — Convenciones y flujo git
- `docs/learning/README.md` — Indice de guias tecnicas
- `docs/learning/yolo_coreml_export.md` — Guia YOLO export
- `docs/learning/depth_anything_export.md` — Guia Depth Anything export

### Verificado
- `uv sync` — 56 packages instalados correctamente
- `uv run python scripts/export_yolo.py` — YOLO26s exportado: 18.4 MB, FP16, 80 clases, end-to-end (NMS-free)
- `uv run python scripts/export_depth.py` — Depth Anything V2 vits exportado: 47.3 MB, FP16, 518x518
- Ambos modelos copiados a `~/Developer/extreme/rayban-nav/RayBanNav/Resources/MLModels/`

### Decisiones tecnicas
- **YOLO26s sobre YOLO11s**: YOLO26 (sept 2025) es la version mas reciente de Ultralytics. Es end-to-end (NMS-free natively), 43% mas rapido en CPU, y optimizado para edge. Output shape `(1, 300, 6)` vs YOLO11 `(1, 84, 8400)`.
- **FP16 por defecto**: Mitad de tamanio, performance similar en Apple Neural Engine. Ambos scripts aceptan `--no-half` para FP32 si se necesita debugging.
- **iOS 18 deployment target**: Para Depth Anything V2, `ct.target.iOS18` permite optimizaciones del compilador CoreML mas recientes.
- **Depth Anything V2 vits (Small)**: 24.8M params, balance entre velocidad y precision. vitb y vitl disponibles via `--variant` pero vits es suficiente para el pipeline de rayban-nav (~15-25 FPS).
- **Scripts standalone**: Cada script es autosuficiente — descarga pesos, clona repos, y exporta. No requieren setup previo mas alla de `uv sync`.
- **YOLO26 NMS warning**: `nms=True` no aplica para modelos end-to-end (YOLO26). Ultralytics fuerza `nms=False` automaticamente. El script pasa `nms=not args.no_nms` pero YOLO26 lo ignora — esto es intencional para mantener compatibilidad con modelos legacy (yolo11, yolov8).
