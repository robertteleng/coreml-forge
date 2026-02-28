# Next Session — MediaPipe Face Landmarker → CoreML (ANE)

**Consumer**: [flow-coach-ios](../flow-coach-ios) — Dance coaching app with real-time pose + hand + face tracking
**Priority**: Face Landmarker first (biggest ANE win), then Hand, then Pose
**Goal**: Replace MediaPipe FaceLandmarker (GPU) with a PyTorch-native face mesh model exported to CoreML for Apple Neural Engine

---

## Context

FlowCoach runs 3 MediaPipe models simultaneously via `MediaPipeTasksVision` SDK:

| Model | File | Landmarks | Current Delegate | FPS Impact |
|-------|------|-----------|-----------------|------------|
| **Pose Landmarker** | `pose_landmarker_full.task` | 33 body | GPU | Base (~25 FPS) |
| **Hand Landmarker** | `hand_landmarker.task` | 21 × 2 hands | GPU | -3 FPS |
| **Face Landmarker** | `face_landmarker.task` | 478 face | GPU | -5 FPS |

All three compete for the same Metal command queue (no parallel GPU streams).
With all 3 active, iPhone Air drops to 16-18 FPS.
The Neural Engine (ANE) sits idle — moving face to ANE frees GPU for pose + hands.

**Benefit:** Face on ANE + Pose/Hand on GPU = true parallelism. No Metal queue contention.

## Why Face First

- Face has 478 landmarks (heaviest model) — biggest GPU pressure
- Face is already frame-skipped (`_faceSkipInterval`) due to perf — not ideal
- Thermal shedding sheds face first (`.serious` thermal state)
- Moving face to ANE would let it run every frame without frame-skipping

## Strategy: PyTorch → CoreML (NOT TFLite)

The TFLite→CoreML route from MediaPipe .task files has high risk of incompatible ops.
The clean route is a **PyTorch-native FaceMesh model** exported with `coremltools.convert()` —
same proven pipeline as YOLO26s and Depth Anything V2 exports in this repo.

### Candidate Models

| Model | Landmarks | Size | PyTorch Native | Export Risk |
|-------|-----------|------|----------------|-------------|
| **MediaPipePyTorch (zmurez)** | 468 | ~3MB | Yes (reimpl.) | Low — same architecture as FaceMesh |
| **facemesh.pytorch (thepowerfuldeez)** | 468 | ~3MB | Yes (reimpl.) | Low |
| face-alignment (Adrian Bulat) | 68 | ~2MB | Yes | Clean — but only 68 landmarks |
| InsightFace 2D106 | 106 | ~5MB | Yes | Clean — well-tested pipeline |

**Recommendation:** `zmurez/MediaPipePyTorch` or `thepowerfuldeez/facemesh.pytorch` — 468 landmarks matching MediaPipe FaceLandmarker indices → **drop-in replacement**. `FacePose` struct unchanged, fusion lines (chin index 152) intact, no re-mapping needed.

### Conversion Pipeline

```
PyTorch FaceMesh model (.pth)
    ↓ torch.jit.trace(model, dummy_input)
TorchScript (.pt)
    ↓ coremltools.convert(traced, ...)
CoreML .mlpackage (FP16, compute_units=CPU_AND_NE)
```

### Export Code (template — adapt from export_depth.py)

```python
import torch
import coremltools as ct

model = load_face_model()  # from zmurez/MediaPipePyTorch
model.eval()
dummy = torch.randn(1, 3, 192, 192)
traced = torch.jit.trace(model, dummy)

mlmodel = ct.convert(
    traced,
    inputs=[ct.ImageType(shape=(1, 3, 192, 192))],
    minimum_deployment_target=ct.target.iOS18,
    compute_precision=ct.precision.FLOAT16,
    compute_units=ct.ComputeUnit.CPU_AND_NE,
)
mlmodel.save("exports/FaceLandmarks.mlpackage")
```

### Key Considerations

1. **Verify landmark indices**: Ensure output indices match MediaPipe's 468-point face mesh (chin=152, forehead, etc.)
2. **Input preprocessing**: Check if model expects RGB normalized [0,1] or [-1,1] — must match CoreML pipeline
3. **Output format**: Map output tensor to `[468 × (x, y, z)]` — FlowCoach adds visibility=1.0 for all face landmarks
4. **ANE compatibility**: Avoid dynamic shapes. Use `ct.ComputeUnit.CPU_AND_NE` — coremltools will warn if ops fall back to CPU
5. **INT8 option**: ANE INT8 is 2x faster than FP16 (proven with DepthAnything export). Consider `ct.optimize.coreml.linear_quantize_weights()` if FP16 insufficient

## Script to Create

```
scripts/export_face_landmarks.py
```

### Interface

```bash
uv run python scripts/export_face_landmarks.py                        # Default: FP16, ANE
uv run python scripts/export_face_landmarks.py --int8                  # INT8 quantization
uv run python scripts/export_face_landmarks.py --variant thepowerfuldeez  # Alt model source
```

### Output

```
exports/FaceLandmarks.mlpackage
```

## Integration Back in FlowCoach

Once exported, the iOS integration path:

1. Add `FaceLandmarks.mlpackage` to Xcode bundle resources
2. Create `CoreMLFaceEstimator` conforming to `FrameConsumer`
3. Use Vision framework `VNCoreMLRequest` for inference on ANE (parallel to GPU-bound MediaPipe)
4. Output → `FacePose(landmarks: [Landmark])` directly (same 468 indices, no re-mapping)
5. Replace MediaPipe `FaceLandmarker` usage in `HolisticEstimator`

## Files in FlowCoach to Reference

- `FlowCoach/Utils/Constants.swift` — model filenames, confidence thresholds
- `FlowCoach/Services/Vision/HolisticEstimator.swift` — current face pipeline (lines 272-300)
- `FlowCoach/Models/Pose/FacePose.swift` — output format (478→468 landmarks)
- `FlowCoach/Models/Pose/Landmark.swift` — `Landmark(x, y, z, visibility)`
- `docs/project/RESEARCH.md` — full research on candidate models (line 221+)

## Success Criteria

- [x] PyTorch FaceMesh model loads and traces cleanly *(2026-02-28: zmurez/MediaPipePyTorch, TraceableFaceLandmark wrapper)*
- [x] CoreML export succeeds with FP16 and `CPU_AND_NE` *(2026-02-28: 1.2 MB, 365 MIL ops)*
- [ ] Output landmarks match MediaPipe 468-point indices (spot-check chin=152, nose tip, eye corners)
- [ ] Model runs on ANE (verify with Instruments → CoreML trace)
- [ ] FP16 inference < 15ms on iPhone 12+ (vs ~20ms current GPU)
- [ ] INT8 variant if FP16 insufficient

## Later: Hand + Pose

Same PyTorch→CoreML pipeline. Priority:
1. **Face** (this session) — biggest win, heaviest model
2. **Hand** — medium win, frees GPU further
3. **Pose** — lowest priority (already fast on GPU as sole model)
