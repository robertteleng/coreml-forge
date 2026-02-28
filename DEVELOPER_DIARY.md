# DEVELOPER_DIARY.md — Bitácora de Diseño de Features

Registro detallado de cómo se pensó y construyó cada feature, aplicando el [Framework de las 5 Preguntas](PAIR_WORKFLOW.md#framework-las-5-preguntas).

Se llena **después** de completar cada feature. El objetivo es ver cómo pensaste, qué decidiste, y por qué — para que el patrón se vuelva natural.

---

## Feature: Face Landmarks → CoreML (ANE)
**Fecha:** 2026-02-28
**Branch:** `main`
**Estado:** Completada (export) — pendiente integración en FlowCoach

---

### P1: Historia del Usuario
> "Yo abro FlowCoach y activo face tracking para coaching de expresión facial durante la danza. Pero con Pose + Hand + Face corriendo todos en GPU, el FPS cae a 16-18 y el face se skipea frames. Quiero que face corra en el Neural Engine para que no compita con los otros modelos y pueda correr cada frame."

### P2: Estados y Transiciones

```
[MediaPipe FaceMesh en GPU]
    ──competencia Metal queue──▶ [16-18 FPS, frame skip]
    ──mover face a ANE──▶ [Face en ANE ∥ Pose+Hand en GPU]
    ──paralelismo real──▶ [~25 FPS, sin frame skip]
```

**¿Por qué estos estados?**
- Los 3 modelos MediaPipe comparten un solo Metal command queue (no hay streams paralelos en Metal)
- Face es el más pesado (478/468 landmarks) y ya se frame-skipea
- ANE está idle — es hardware dedicado que puede correr en paralelo al GPU

**¿Qué descarté?**
- TFLite → CoreML: los `.task` de MediaPipe tienen custom ops incompatibles
- Modelos con menos landmarks (face-alignment 68pts, InsightFace 106pts): no son drop-in replacement
- `face_landmark_with_attention.tflite` (478 landmarks): custom TFLite ops, no convertible a ningún framework

### P3: Veo / Necesito

| Estado | Lo que ve el usuario | Datos que necesito | De dónde vienen |
|---|---|---|---|
| Pre-export | Script no existe | PyTorch FaceMesh model, pesos, pipeline probada | GitHub (zmurez/MediaPipePyTorch) |
| Export | `.mlpackage` generado | Trace sin errores, output shape correcto | `torch.jit.trace` → `coremltools.convert` |
| Post-export | Modelo listo para iOS | Verificar landmark indices, tamaño, precision | Spot-check en output |

### P4: Inventario

| Necesito | ¿Existe? | Decisión | Por qué |
|---|---|---|---|
| Pipeline PyTorch→CoreML | Sí (`export_depth.py`) | Reusar patrón | Misma estructura: clone repo → load → trace → convert → save |
| Modelo PyTorch FaceMesh | Sí (zmurez) | Reusar | 468 landmarks, pesos incluidos, forward() limpio |
| Wrapper traceable | No | Crear | Original tiene `if x.shape[0] == 0` que rompe tracing |
| INT8 quantization | Sí (coremltools) | Reusar | `linear_quantize_weights()` ya probado |

### P5: Diagrama de Pegamento

```
[zmurez/MediaPipePyTorch]
    ──git clone──▶ [load_model()]
    ──crear wrapper──▶ [create_traceable_model()]
    ──torch.jit.trace──▶ [TorchScript]
    ──coremltools.convert──▶ [FaceLandmarks.mlpackage]
    ──cp a Xcode──▶ [FlowCoach: VNCoreMLRequest en ANE]
```

**¿Por qué esta conexión?**
- Es el mismo pipeline probado con YOLO y Depth Anything — bajo riesgo
- El wrapper es necesario porque `torch.jit.trace` no soporta control flow dinámico

### Implementación

**Archivos creados:**
- `scripts/export_face_landmarks.py` — script completo de export

**Archivos modificados:**
- `pyproject.toml` — agregado `forge-face` script entry
- `CLAUDE.md` — agregado Face Landmarks a tabla de modelos, quick commands, estructura

**Decisiones clave:**
1. **zmurez/MediaPipePyTorch sobre thepowerfuldeez**: menor error reportado, normalización más limpia (0-1 vs -1 a 1), output ya reshape a (batch, 468, 3)
2. **Wrapper `TraceableFaceLandmark`**: copia backbone1/2a/2b del modelo original y replica el forward() sin el guard `if x.shape[0] == 0`. Usa `reshape(1, 468, 3)` en vez de `view(-1, 468, 3)` para compatibilidad con tracing
3. **`ct.ComputeUnit.CPU_AND_NE`**: le dice a CoreML que puede usar ANE. Sin dynamic shapes para máxima compatibilidad ANE
4. **Normalización `scale=1/255.0`**: el modelo original divide por 255 (rango 0-1), no por 127.5. Esto se configura en `ct.ImageType` para que CoreML lo haga automáticamente
5. **Output: (landmarks, confidence)**: se exportan ambos como outputs nombrados para que Swift pueda acceder `landmarks` directamente

**Modelo resultante:**
- **Tamaño**: 1.2 MB (vs ~3 MB PyTorch original — FP16 compression)
- **Input**: 192x192 RGB image (CoreML maneja normalización)
- **Output**: `landmarks` [1, 468, 3] (x,y,z normalized) + `confidence` [1] (sigmoid)
- **365 ops** convertidas a MIL graph

### Reflexión
- **Lo que funcionó:** El pipeline PyTorch→CoreML está tan probado en este repo que adaptar `export_depth.py` fue directo. El modelo de zmurez es clean — pesos incluidos, forward corto.
- **Lo que costó:** Investigar qué modelo usar. Hay muchos repos de FaceMesh PyTorch pero la mayoría tienen issues de accuracy por diferencias de padding entre TFLite y PyTorch. El modelo de 478 landmarks (con attention) no es convertible — dead end confirmado.
- **Lo que haría diferente:** Empezar validando accuracy del modelo PyTorch vs MediaPipe nativo antes de exportar. Ahora tenemos el .mlpackage pero falta verificar que los landmarks realmente coinciden con spot-checks en device.
- **Patrón reutilizable:** `create_traceable_model()` — cuando un modelo tiene control flow dinámico en forward(), crear un wrapper que copia los submodules y replica el forward sin los guards. Funciona para cualquier modelo con `if batch_size == 0` patterns.

---

## Próximos Pasos — Integración en FlowCoach

El modelo está exportado. Ahora toca integrarlo en [flow-coach-ios](../flow-coach-ios). Ver sección abajo en detalle.

---

## Plantilla por Feature

Copia esto para cada feature nueva:

```markdown
## Feature: [nombre]
**Fecha:** YYYY-MM-DD
**Branch:** `feature/...`
**Estado:** Completada / En progreso

---

### P1: Historia del Usuario
> "Yo [acción en primera persona]..."

### P2: Estados y Transiciones

¿Cuándo cambia algo? Dibuja cajas con flechas:
```
[A] ──qué pasa──▶ [B] ──qué pasa──▶ [C]
```

**¿Por qué estos estados?**
- ...

**¿Qué descarté?**
- ...

### P3: Veo / Necesito

| Estado | Lo que ve el usuario | Datos que necesito | De dónde vienen |
|---|---|---|---|
| ... | ... | ... | ... |

### P4: Inventario

| Necesito | ¿Existe? | Decisión | Por qué |
|---|---|---|---|
| ... | ... | Reusar / Extender / Crear | ... |

### P5: Diagrama de Pegamento

```
[Servicio A] ──evento──▶ [pegamento] ──acción──▶ [Servicio B]
```

**¿Por qué esta conexión?**
- ...

### Implementación

**Archivos creados:**
- ...

**Archivos modificados:**
- ...

**Decisiones clave:**
- ...

**Diagrama de arquitectura final:**
(cómo quedaron conectados los componentes)

### Reflexión
- **Lo que funcionó:** ...
- **Lo que costó:** ...
- **Lo que haría diferente:** ...
- **Patrón reutilizable:** ...
```
