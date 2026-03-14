# Whisper → CoreML Export: Learnings

## Por qué solo el encoder va a CoreML

Whisper tiene encoder (mel → features) y decoder (features + tokens → next token).

**El decoder no se puede exportar limpiamente a CoreML:**
- `torch.jit.trace` captura la gráfica para un input fijo. Si trazas con `tokens=(1,1)`, las positional embeddings quedan fijas para posición 0.
- `ct.RangeDim` permite shapes variables en la conversión CoreML, pero NO arregla un trace que ya está fijo.
- El decoder tiene cross-attention con KV cache que tampoco se traza bien.

**Solución**: Encoder a CoreML (es el compute pesado, ~95% del FLOP). Decoder en Swift con Accelerate.framework. Apple hace lo mismo en su proyecto ml-stable-diffusion.

## Qué exportamos

```
exports/Whisper_{variant}/
├── WhisperEncoder_{variant}.mlpackage   # CoreML (ANE)
├── decoder_weights.safetensors          # Pesos para Swift
├── tokenizer.json                       # BPE vocab + special tokens
└── config.json                          # Dimensiones del modelo
```

## Encoder: trace limpio

El encoder no tiene control flow dinámico. Input fijo `(1, 80, 3000)` → output `(1, 1500, d_model)`.

```python
mel = torch.randn(1, 80, 3000)  # 30s a 16kHz
traced = torch.jit.trace(encoder, mel)  # Sin problemas
```

## Mel spectrogram en Swift

NO exportamos el mel spectrogram como CoreML. Se computa en Swift con Accelerate:

| Parámetro | Valor |
|-----------|-------|
| Sample rate | 16000 Hz |
| FFT size | 400 |
| Hop length | 160 |
| Mel bins | 80 |
| Chunk | 30 segundos (480000 samples → 3000 frames) |

## Decoder en Swift

El decoder es un transformer estándar:
1. Token embedding + learned positional embedding
2. N bloques con self-attention + cross-attention (al encoder output) + FFN
3. Proyección lineal → logits (vocab size)

Loop autoregresivo:
```
tokens = [SOT]
while tokens[-1] != EOT and len(tokens) < 448:
    logits = decoder(tokens, encoder_features)
    next_token = argmax(logits[-1])  # greedy
    tokens.append(next_token)
```

Los pesos se cargan desde `decoder_weights.safetensors` (float16).

## Variantes

| Variante | Encoder size | Decoder weights | d_model | Layers |
|----------|-------------|----------------|---------|--------|
| tiny | ~15 MB | ~56 MB | 384 | 4 |
| base | ~30 MB | ~112 MB | 512 | 6 |
| small | ~230 MB | ~230 MB | 768 | 12 |
| medium | ~750 MB | ~750 MB | 1024 | 24 |

## Tokenizer

Whisper usa BPE (tiktoken). Exportamos el vocab como JSON con:
- `vocab`: mapa token_string → rank
- `special_tokens`: SOT (50258), EOT (50257), timestamps, language tokens, etc.

En Swift: implementar BPE decode con el vocab. No se necesita encode (los tokens de input son solo special tokens controlados por el decoder loop).
