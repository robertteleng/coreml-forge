# Next Session — MediaPipe Face Landmarker → CoreML (ANE)

**Consumer**: [flow-coach-ios](../flow-coach-ios) — Dance coaching app with real-time pose + hand + face tracking
**Priority**: Face Landmarker first (biggest ANE win), then Hand, then Pose
**Goal**: Convert MediaPipe TFLite models to CoreML so they run on Apple Neural Engine instead of GPU

---

## Context

FlowCoach runs 3 MediaPipe models simultaneously via `MediaPipeTasksVision` SDK:

| Model | File | Landmarks | Current Delegate | FPS Impact |
|-------|------|-----------|-----------------|------------|
| **Pose Landmarker** | `pose_landmarker_full.task` | 33 body | GPU | Base (~25 FPS) |
| **Hand Landmarker** | `hand_landmarker.task` | 21 × 2 hands | GPU | -3 FPS |
| **Face Landmarker** | `face_landmarker.task` | 478 face | GPU | -5 FPS |

All three compete for the same GPU. With all 3 active, iPhone Air drops to 16-18 FPS.
The Neural Engine (ANE) sits idle — moving face (and ideally hand) to ANE would free GPU for pose.

## Why Face First

- Face has 478 landmarks (heaviest model) — biggest GPU pressure
- Face is already frame-skipped (`_faceSkipInterval`) due to perf — not ideal
- Thermal shedding sheds face first (`.serious` thermal state)
- Moving face to ANE would let it run every frame without frame-skipping

## Source Models

MediaPipe `.task` files are actually ZIP archives containing TFLite flatbuffers:

```bash
# Extract the TFLite model from .task file
unzip -o face_landmarker.task -d face_landmarker_extracted/
# Look for .tflite files inside
find face_landmarker_extracted/ -name "*.tflite"
```

The `.task` bundle may contain multiple sub-models (detector + landmarker). Each needs separate conversion.

## Conversion Pipeline

```
MediaPipe .task (ZIP)
    ↓ unzip
TFLite .tflite (FlatBuffer)
    ↓ tflite2tensorflow or ai-edge-torch
TensorFlow SavedModel / ONNX
    ↓ coremltools
CoreML .mlpackage (FP16, ANE-optimized)
```

### Tools

```bash
# Option A: TFLite → TF SavedModel → CoreML
pip install tflite2tensorflow coremltools

# Option B: TFLite → ONNX → CoreML
pip install tf2onnx onnx coremltools

# Option C: Direct (if coremltools supports the ops)
pip install coremltools  # ct.converters.convert() with source="tensorflow"
```

### Key Considerations

1. **Multi-model bundle**: The .task file likely contains a face detector + face mesh model. Both need conversion
2. **Input preprocessing**: MediaPipe applies specific normalization — must match in CoreML pipeline
3. **Output format**: Need to map TFLite output tensors to the same landmark format (`[478 × (x,y,z,visibility)]`)
4. **ANE compatibility**: Avoid ops that fall back to CPU (dynamic shapes, certain activations). Use `ct.ComputeUnit.CPU_AND_NE` for testing
5. **Quantization**: INT8 on ANE is 2x faster than FP16 (see depth export experience). Consider post-training quantization

## Script to Create

```
scripts/export_mediapipe_face.py
```

### Interface

```bash
uv run python scripts/export_mediapipe_face.py                    # Default: FP16, ANE
uv run python scripts/export_mediapipe_face.py --int8             # INT8 quantization
uv run python scripts/export_mediapipe_face.py --task path/to.task  # Custom .task file
```

### Output

```
exports/FaceLandmarker.mlpackage
```

## Integration Back in FlowCoach

Once converted, the FlowCoach integration path:

1. Add `FaceLandmarker.mlpackage` to Xcode bundle resources
2. Create `CoreMLFaceEstimator` conforming to `FrameConsumer`
3. Use Vision framework `VNCoreMLRequest` for inference
4. Map output to existing `FacePose(landmarks: [Landmark])` struct
5. Replace MediaPipe `FaceLandmarker` usage in `HolisticEstimator`
6. Set `VNCoreMLRequest.preferredMetalDevice = nil` to force ANE

## Files in FlowCoach to Reference

- `FlowCoach/Utils/Constants.swift` — model filenames, confidence thresholds
- `FlowCoach/Services/Vision/HolisticEstimator.swift` — current face pipeline
- `FlowCoach/Models/Pose/FacePose.swift` — output format (478 landmarks)
- `FlowCoach/Models/Pose/Landmark.swift` — `Landmark(x, y, z, visibility)`

## Success Criteria

- [ ] Face .tflite extracted and converted to .mlpackage
- [ ] Model runs on ANE (verify with Instruments → CoreML trace)
- [ ] Output matches MediaPipe landmark format (478 points, same coordinate space)
- [ ] FP16 inference < 15ms on iPhone 12+ (vs ~20ms current GPU)
- [ ] INT8 variant if FP16 insufficient

## Later: Hand + Pose

Same pipeline applies. Priority order:
1. **Face** (this session) — biggest win, heaviest model
2. **Hand** — medium win, frees GPU further
3. **Pose** — lowest priority (already fast on GPU as sole model)
