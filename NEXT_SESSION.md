# Next Session — Re-exportar Whisper encoder para ANE (iOS 26)

## Problema

El `.mlpackage` actual usa formato de pesos v1 que iOS 26 no acepta en ANE.
Error en device: `Storage Reader expects file format version 2` y `BNNS failed to compile`.

## Solución

Re-exportar con `minimum_deployment_target=ct.target.iOS18` para forzar formato v2.

```bash
cd /path/to/coreml-forge
source .venv/bin/activate

python3 -c "
import coremltools as ct
import torch
from transformers import WhisperForConditionalGeneration

# Cargar modelo (ya cacheado en HuggingFace)
model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-large-v3-turbo')
encoder = model.get_encoder()
encoder.eval()

# Trace con input dummy
mel_input = torch.randn(1, 128, 3000)
traced = torch.jit.trace(encoder, mel_input)

# Convertir a Core ML
mlmodel = ct.convert(
    traced,
    inputs=[ct.TensorType(name='mel_spectrogram', shape=(1, 128, 3000))],
    outputs=[ct.TensorType(name='audio_features')],
    compute_precision=ct.precision.FLOAT16,
    minimum_deployment_target=ct.target.iOS18,
    compute_units=ct.ComputeUnit.ALL,
)

mlmodel.save('exports/Whisper_large_v3_turbo/WhisperEncoder_large_v3_turbo.mlpackage')
print('Done!')
"
```

## Después de exportar

1. Compilar para iOS:
   ```bash
   xcrun coremlcompiler compile \
     exports/Whisper_large_v3_turbo/WhisperEncoder_large_v3_turbo.mlpackage \
     exports/Whisper_large_v3_turbo/ \
     --target ios
   ```

2. Copiar el `.mlmodelc` al repo de brevox-ios:
   ```bash
   cp -R exports/Whisper_large_v3_turbo/WhisperEncoder_large_v3_turbo.mlmodelc \
     /path/to/brevox-ios/Brevox/Resources/Models/Whisper_large_v3_turbo.bundle/
   ```

3. En `brevox-ios`, editar `CoreMLWhisperTranscriber.swift:65`:
   ```swift
   // Cambiar:
   encoderConfig.computeUnits = .cpuAndGPU
   // Por:
   encoderConfig.computeUnits = .all
   ```

4. Clean build (Cmd+Shift+K) + Run en iPhone (Cmd+R)

## Verificación

- Log: `Encoder loaded on ANE` sin errores `E5RT` ni `BNNS`
- Transcripción real (no "Local transcript for UUID...")
- Sin crash de memoria (Whisper large-v3-turbo ~600 MB en ANE)
