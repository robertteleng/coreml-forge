# Changelog — coreml-forge

Registro cronologico de desarrollo, decisiones tecnicas, y cambios significativos.

---

## 2026-03-15 — Whisper large-v3-turbo Export

### Creado

- `scripts/export_whisper_turbo.py` — Export whisper-large-v3-turbo via HF `transformers` (no `openai-whisper`). Encoder a CoreML + decoder weights como safetensors.

### Modificado

- `pyproject.toml` — Agregado `transformers>=4.44` + entry point `forge-whisper-turbo`
- `CLAUDE.md` — Agregado Whisper large-v3-turbo a tabla de modelos, commands, estructura
- `docs/project/IMPLEMENTATION_PLAN.md` — Actualizado progreso Whisper, modelo actual es large-v3-turbo
- `docs/project/NEXT_SESSION.md` — Marcado task 1 completado, agregado task de integración en Brevox

### Decisiones técnicas

- **HF transformers sobre openai-whisper**: `large-v3-turbo` solo está disponible en HuggingFace, no en el paquete `openai-whisper` que solo soporta tiny/base/small/medium.
- **128 mel bins**: large-v3-turbo usa 128 (vs 80 en small). Brevox necesita actualizar su computación de mel.
- **EncoderWrapper**: El encoder de HF espera `input_features` como kwarg; el wrapper convierte a positional arg para que `torch.jit.trace` funcione.
- **proj_out incluido**: Los pesos del decoder incluyen `lm_head` (proyección a vocabulario) que en el modelo HF se llama `proj_out`.
- **Tamaños**: Encoder 1215 MB, decoder 455 MB — significativamente más grande que small pero con 4 decoder layers el inference es rápido.

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
- **YOLO26s**: YOLO26 (sept 2025) es la version mas reciente de Ultralytics. End-to-end (NMS-free), 43% mas rapido en CPU, optimizado para edge. Output shape `(1, 300, 6)`.
- **FP16 por defecto**: Mitad de tamanio, performance similar en Apple Neural Engine. Ambos scripts aceptan `--no-half` para FP32 si se necesita debugging.
- **iOS 18 deployment target**: Para Depth Anything V2, `ct.target.iOS18` permite optimizaciones del compilador CoreML mas recientes.
- **Depth Anything V2 vits (Small)**: 24.8M params, balance entre velocidad y precision. vitb y vitl disponibles via `--variant` pero vits es suficiente para el pipeline de rayban-nav (~15-25 FPS).
- **Scripts standalone**: Cada script es autosuficiente — descarga pesos, clona repos, y exporta. No requieren setup previo mas alla de `uv sync`.
