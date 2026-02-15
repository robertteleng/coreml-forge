# Workflow — coreml-forge

Convenciones, flujo de trabajo, y reglas para contribuir.

---

## Nomenclatura

| Elemento | Convencion | Ejemplo |
|----------|-----------|---------|
| Scripts | snake_case | `export_yolo.py`, `export_depth.py` |
| Funciones | snake_case | `ensure_depth_anything_repo()` |
| Variables | snake_case | `output_dir`, `weights_path` |
| Constantes (dict) | SCREAMING_SNAKE_CASE | `VARIANTS` |
| CLI args | kebab-case | `--no-half`, `--output-dir` |
| Output files | PascalCase para modelos | `DepthAnythingV2_vits.mlpackage` |

---

## Estructura de Scripts

```python
"""Docstring con Usage examples y Output path."""

import argparse
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()

def main():
    # 1. Parse args con defaults sensatos
    parser = argparse.ArgumentParser(...)
    args = parser.parse_args()

    # 2. Print config table (Rich)
    table = Table(title="Export Config")
    ...

    # 3. Import heavy libs aqui (no al top — para que --help sea rapido)
    from ultralytics import YOLO

    # 4. Load model
    # 5. Export / Convert
    # 6. Move to output dir
    # 7. Print summary table (size, input shape, precision)
    # 8. Print next step (cp command)

if __name__ == "__main__":
    main()
```

### Reglas de scripts
- **`--help` rapido**: Heavy imports (torch, ultralytics, coremltools) dentro de `main()`, no al top
- **Defaults sensatos**: Siempre funciona sin argumentos
- **Rich output**: Tablas de config y summary con `rich`
- **Print model info**: Tamano, input shape, precision — siempre
- **Print next step**: Comando `cp` para copiar a rayban-nav

---

## Patrones

### DO
- `uv` para todo (sync, run)
- `argparse` con `--help` descriptivo
- `Path` sobre strings para rutas
- FP16 por defecto (Neural Engine optimizado)
- iOS 18 deployment target
- Descargar weights automaticamente (HuggingFace, Ultralytics)

### DO NOT
- `pip install` o `conda`
- Hard-code rutas absolutas en scripts
- Commit modelos binarios (.mlpackage, .pt, .pth)
- Export sin print de model info
- Dependencies innecesarias

---

## Git Flow

### Formato de Commits

```
type(scope): short description
```

**Types**: `feat`, `fix`, `docs`, `chore`, `perf`

**Scopes**: `yolo`, `depth`, `scripts`, `config`

**Ejemplos**:
```
feat(yolo): add yolo26s export with end-to-end NMS
feat(depth): add Depth Anything V2 vits export
fix(depth): handle torch.load weights_only deprecation
docs(learning): add YOLO CoreML export guide
chore(config): update ultralytics to 8.5
perf(yolo): export with INT8 quantization
```

### Reglas
- Commits pequenos y frecuentes
- Un cambio logico por commit
- No Co-Authored-By trailers
- No force push a main

---

## Dependencias

### Aprobadas
- **ultralytics** — YOLO model loading + export
- **coremltools** — CoreML conversion
- **torch** + **torchvision** — PyTorch runtime
- **huggingface-hub** — Download model weights
- **rich** — Terminal output formatting

### Gestion
- Todo via `pyproject.toml` + `uv sync`
- Pin major versions en pyproject.toml (`>=8.3`, `>=8.0`, etc.)
- `uv.lock` gitignored (regenerable)

---

## Testing de Exports

```bash
# Exportar y verificar
uv run python scripts/export_yolo.py
uv run python scripts/export_depth.py

# Copiar a rayban-nav
cp -r exports/*.mlpackage ~/Developer/extreme/rayban-nav/RayBanNav/Resources/MLModels/

# Verificar en Xcode: Build rayban-nav, check que los modelos compilan
```

No hay unit tests — la verificacion es que el export complete sin errores y el modelo cargue en Xcode.
