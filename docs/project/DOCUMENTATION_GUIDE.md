# Guia de Documentacion — coreml-forge

Guia general para entender el proyecto. Explica que es cada archivo, para que sirve, y en que orden leerlos.

---

## Estructura General

```
coreml-forge/
├── CLAUDE.md                          # Reglas para LLM / agentes AI
├── README.md                          # Vision del proyecto, setup, uso
├── pyproject.toml                     # Config uv + dependencias
├── docs/
│   ├── project/
│   │   ├── DOCUMENTATION_GUIDE.md     # ← Este archivo
│   │   ├── CHANGELOG.md              # Historial de desarrollo
│   │   ├── IMPLEMENTATION_PLAN.md    # Roadmap de modelos a exportar
│   │   └── WORKFLOW.md               # Convenciones y flujo de trabajo
│   └── learning/
│       ├── README.md                  # Indice de guias tecnicas
│       ├── yolo_coreml_export.md      # YOLO → CoreML: opciones, tradeoffs
│       └── depth_anything_export.md   # Depth Anything V2 → CoreML
├── scripts/
│   ├── export_yolo.py                 # YOLO → .mlpackage
│   └── export_depth.py               # Depth Anything V2 → .mlpackage
└── exports/                           # Output .mlpackage (gitignored)
```

---

## Archivos Raiz

### CLAUDE.md
**Para quien:** LLMs, agentes AI, y devs.

Contiene todo lo que necesitas saber antes de escribir codigo:
- Identidad del proyecto y consumidor (rayban-nav)
- Quick commands (`uv sync`, export, copy)
- Estructura de archivos
- Reglas (uv only, FP16 default, iOS 18, print model info)

**Cuando leerlo:** Siempre. Es lo primero.

### README.md
**Para quien:** Cualquiera que quiera usar coreml-forge.

Contiene:
- Que es y por que existe
- Setup rapido con uv
- Uso de cada script con ejemplos
- Tabla de modelos soportados con specs
- Referencia cruzada con rayban-nav

**Cuando leerlo:** Al inicio.

---

## docs/project/ — Planificacion y Proceso

### DOCUMENTATION_GUIDE.md (este archivo)
Mapa de todos los documentos y orden de lectura.

### CHANGELOG.md
Registro cronologico de desarrollo, decisiones tecnicas, y cambios.

**Cuando leerlo:** Para entender por que algo esta como esta.

### IMPLEMENTATION_PLAN.md
Roadmap de modelos a exportar y features por implementar.

**Cuando leerlo:** Para saber que esta hecho y que falta.

### WORKFLOW.md
Convenciones de codigo, formato de commits, flujo git.

**Cuando leerlo:** Antes de tu primer commit.

---

## docs/learning/ — Guias Tecnicas

### learning/README.md
Indice de guias de exportacion.

### learning/yolo_coreml_export.md
Todo sobre exportar YOLO a CoreML: variantes, NMS, precision, input sizes, performance en iPhone.

### learning/depth_anything_export.md
Depth Anything V2 a CoreML: variantes (vits/vitb/vitl), tracing, conversion, profundidad inversa.

**Cuando leerlos:** Antes de modificar los scripts de export o agregar nuevos modelos.

---

## Mapeo coreml-forge → rayban-nav

| coreml-forge | rayban-nav | Notas |
|---|---|---|
| `exports/yolo26s.mlpackage` | `Resources/MLModels/yolo26s.mlpackage` | `cp -r` al proyecto Xcode |
| `exports/DepthAnythingV2_vits.mlpackage` | `Resources/MLModels/DepthAnythingV2_vits.mlpackage` | Depth estimation monocular |
| `scripts/export_yolo.py` | `DetectorService.swift` | El modelo que exportamos lo consume DetectorService |
| `scripts/export_depth.py` | `DetectorService.swift` | Pipeline paralelo YOLO + Depth |

---

## Orden de Lectura Recomendado

### Si eres nuevo:
1. **CLAUDE.md** — Reglas y quick commands (2 min)
2. **README.md** — Que es y como usarlo (3 min)
3. **IMPLEMENTATION_PLAN.md** — Que modelos exportar (2 min)

### Si vas a modificar scripts:
4. **WORKFLOW.md** — Convenciones (2 min)
5. **learning/** — Guias tecnicas del modelo que vas a tocar

### Si quieres entender decisiones pasadas:
- **CHANGELOG.md** — Historia del proyecto

---

## Principios de Documentacion

1. **CLAUDE.md como fuente de verdad** — Un solo archivo con todas las reglas
2. **README.md para el "que"** — Que es, como usarlo, tabla de modelos
3. **Plan separado del proceso** — IMPLEMENTATION_PLAN (que exportar) vs WORKFLOW (como contribuir)
4. **Historia en un solo lugar** — CHANGELOG consolida todo
5. **Guias tecnicas por modelo** — Separar por tipo de modelo, no por fase
6. **Referencia cruzada con rayban-nav** — Siempre documentar que modelo va a donde
7. **Consolidar agresivamente** — Menos archivos > mas archivos
