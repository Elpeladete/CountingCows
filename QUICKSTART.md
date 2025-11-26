# 🚀 Inicio Rápido - AgriSense AI

## ✅ Estado de Instalación

- [x] Python 3.12.10 instalado
- [x] Entorno virtual creado (`venv/`)
- [x] Dependencias básicas instaladas:
  - NumPy 2.2.6
  - OpenCV 4.12.0
  - PyYAML, Pillow, SciPy, Matplotlib
  - Requests, Loguru, Pytest
- [x] Estructura de directorios creada

## 📂 Estructura Actual

```
CountingCows/
├── src/                    # Código fuente
│   ├── detector.py        # Sistema de detección YOLOv8
│   ├── counter.py         # Conteo con tracking
│   └── cloud_api.py       # Integración DJI FlightHub 2
├── scripts/               # Scripts de utilidad
│   ├── train_model.py    # Entrenamiento
│   ├── demo.py           # Demo completo
│   └── setup_initial.py  # Setup inicial
├── configs/              # Configuración
│   └── config.yaml       # Config principal
├── datasets/             # Datasets (vacío)
├── models/               # Modelos (vacío)
├── results/              # Resultados
├── logs/                 # Logs
└── venv/                 # Entorno virtual ✅
```

## 🎯 Qué Hacer Ahora

### 1️⃣ Ejecutar el Demo (Recomendado)

```bash
python scripts/demo.py --full
```

Esto te mostrará cómo funciona el sistema sin necesidad de un modelo real.

### 2️⃣ Configurar Credenciales DJI

Edita `configs/config.yaml`:

```yaml
dji_cloud:
  app_id: "TU_APP_ID"           # ← Cambiar
  app_key: "TU_APP_KEY"         # ← Cambiar
  app_license: "TU_LICENSE"     # ← Cambiar
  workspace_id: "TU_WORKSPACE"  # ← Cambiar
```

### 3️⃣ Preparar Dataset (Si vas a entrenar)

```bash
# Ver instrucciones
python scripts/train_model.py --prepare-dataset

# Luego coloca tus imágenes en:
# datasets/coco/train2017/
# datasets/coco/annotations/instances_train2017.json
```

### 4️⃣ Instalar MMYOLO (Para entrenamiento)

```bash
python scripts/train_model.py --setup
```

Sigue las instrucciones que aparecen en pantalla.

## 🧪 Comandos Útiles

### Verificar Instalación
```bash
python -c "import numpy; import cv2; print('OK')"
```

### Ejecutar Demo Específico
```bash
python scripts/demo.py --detection  # Solo detección
python scripts/demo.py --cloud      # Solo Cloud API
python scripts/demo.py --thermal    # Solo térmico
```

### Procesar Imagen (cuando tengas modelo)
```bash
python main.py --image foto.jpg --gps 40.4168 -3.7038
```

### Procesar Video (cuando tengas modelo)
```bash
python main.py --video video.mp4 --report
```

## ⚠️ Importante

1. **El sistema funciona SIN modelo** - Puedes ejecutar el demo para ver cómo funcionará
2. **Para producción necesitas**:
   - Entrenar modelo YOLOv8 con tu dataset
   - Cuantificar el modelo con DJI
   - Instalar en DJI Pilot 2

3. **El modelo debe ser cuantificado por DJI** - No puedes usar directamente un .pth

## 📚 Documentación Completa

- `README.md` - Documentación del proyecto
- `USAGE_GUIDE.md` - Guía detallada de uso
- `scripts/train_model.py --show-instructions` - Comandos de entrenamiento

## 🆘 Soporte

Si tienes problemas:
1. Verifica que el entorno virtual esté activo: `(venv)` en el prompt
2. Ejecuta: `python scripts/setup_initial.py`
3. Revisa `USAGE_GUIDE.md` para más detalles

---

**Siguiente paso recomendado**: `python scripts/demo.py --full`
