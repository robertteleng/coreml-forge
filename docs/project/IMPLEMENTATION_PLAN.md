# Plan de Implementación — coreml-forge

Modelos ML exportados a CoreML para apps iOS. Organizado por consumer.

---

## Consumer: rayban-nav (Visión)

**App**: [rayban-nav](https://github.com/robertteleng/rayban-nav) — Detección + profundidad en tiempo real

### Fase 1: Modelos Core (completada)

#### Hito 1.1 — YOLO Object Detection
- [x] Script `export_yolo.py` con CLI completo (`--model`, `--imgsz`, `--no-nms`, `--no-half`)
- [x] Default: YOLO26s, 640x640, FP16
- [x] Export verificado: 18.4 MB, 80 clases, end-to-end (NMS-free)
- [x] Copiado a rayban-nav `Resources/MLModels/`

#### Hito 1.2 — Depth Estimation
- [x] Script `export_depth.py` con CLI completo (`--variant`, `--imgsz`, `--no-half`)
- [x] Default: vits (Small), 518x518, FP16
- [x] Auto-clone de repo Depth Anything V2 + descarga de weights
- [x] Export verificado: 47.3 MB, depth map output
- [x] Copiado a rayban-nav `Resources/MLModels/`

### Fase 2: Optimización (pendiente)

- [ ] Exportar YOLO26n (Nano) — más rápido, menos preciso
- [ ] Exportar Depth Anything V2 con imgsz 256 — más rápido
- [ ] Evaluar INT8 quantization para ambos modelos
- [ ] Benchmark: comparar FPS y precisión vs variantes Small

---

## Consumer: flow-coach-ios (Face/Body)

**App**: [flow-coach-ios](../../flow-coach-ios) — Dance coaching con tracking de cara, manos y pose

### Fase 1: Face Landmarks (completada)

- [x] Script `export_face_landmarks.py` (468 puntos, MediaPipe-compatible)
- [x] PyTorch FaceMesh (zmurez/MediaPipePyTorch) → CoreML FP16 + ANE
- [x] Export: 1.2 MB, CPU_AND_NE
- [ ] Verificar landmark indices vs MediaPipe nativo
- [ ] Verificar ejecución en ANE con Instruments

### Fase 2: Hand + Pose (pendiente)

- [ ] Mismo pipeline PyTorch→CoreML
- [ ] Menor prioridad que Brevox

---

## Consumer: brevox-ios (Audio/Lenguaje)

**App**: [brevox-ios](../../apps/brevox-ios) — Grabador de voz con transcripción + resumen local
**Dispositivo mínimo**: iPhone con 12GB RAM (15 Pro Max, 16 Pro/Max)
**Branch**: `feat/whisper-export`

### Fase 1: Whisper Transcription (en progreso)

**Modelo**: whisper-small (244M params, ~460 MB FP16)

**Arquitectura de export**: Solo encoder a CoreML, decoder en Swift
- El decoder de Whisper no se puede trazar limpiamente (`torch.jit.trace` fija las positional embeddings)
- El encoder es el compute pesado (corre una vez por chunk de 30s) → CoreML/ANE
- El decoder se implementa en Swift con Accelerate (transformer estándar)

**Entregables**:
```
exports/Whisper_{variant}/
├── WhisperEncoder_{variant}.mlpackage   # CoreML encoder (ANE)
├── decoder_weights.safetensors          # Pesos del decoder para Swift
├── tokenizer.json                       # Vocabulario BPE
└── config.json                          # Dimensiones del modelo
```

**Progreso**:
- [x] Script `export_whisper.py` con encoder CoreML + decoder weights + tokenizer + config
- [x] Testeado con whisper-tiny (15.7 MB encoder, 56.4 MB decoder weights)
- [x] Exportar whisper-small (variante de producción, 168 MB encoder + 293 MB decoder)
- [ ] Verificar encoder output en Xcode
- [ ] Implementar decoder en Swift (brevox-ios)

### Fase 2: Summarizer (en progreso)

**Modelo**: Qwen3.5-4B (via ANEMLL → CoreML/ANE)

**Por qué Qwen3.5-4B**: Modelo más reciente (feb 2026), soportado por ANEMLL para ANE nativo, calidad que rivaliza con modelos mucho mayores. ANEMLL no soporta Phi-4-mini.

**Conversión**: ANEMLL `convert_model.sh` (requiere macOS + Apple Silicon)
- Quantización: LUT4 (FFN) + LUT6 (LM head) → ~2.5 GB
- Formato: CoreML chunked (meta.yaml + mlpackage chunks)

**Entregables**:
```
exports/Summarizer_Qwen3.5_4B/
├── meta.yaml               # ANEMLL config
├── embeddings/             # CoreML
├── lm_head/                # CoreML
└── ffn_*/                  # CoreML chunks
```

**Progreso**:
- [x] Script `export_summarizer.py` (wrapper ANEMLL, macOS only)
- [ ] Ejecutar conversión en Mac
- [ ] Test con ANEMLL chat.py
- [ ] Verificar JSON output (summary, highlights, actionItems)

### Integración en Brevox

**Whisper**:
1. Copiar `Whisper_{variant}/` a Xcode bundle
2. Mel spectrogram en Swift (Accelerate: 80 mel bins, 16kHz, hop=160, FFT=400)
3. Encoder CoreML → decoder Swift (greedy/beam search)
4. Mapear tokens → `TranscriptSegment(start, end, text)`

**Summarizer**:
1. Qwen3.5-4B via Background Assets (~2.5 GB, on-demand download)
2. `CoreMLSummarizationEngine: SummarizationEngine`
3. Mantener `HeuristicSummarizationEngine` como fallback

**Archivos a modificar en brevox-ios**:
- `CoreMLWhisperTranscriber.swift` — pipeline real
- `SummarizationEngine.swift` — nuevo engine CoreML
- `project.yml` — recursos de modelo
- Futuro (iOS 26): migrar a Apple Foundation Models

---

## Notas Generales

- **uv** para todo (no pip, no conda)
- Exports van a `exports/` (gitignored)
- Clones de repos van a `.models/` (gitignored)
- Scripts standalone con `--help` y defaults sensatos
- Default: FP16, iOS 18 deployment target
- Commits: `type(scope): description`
