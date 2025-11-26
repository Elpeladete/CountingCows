# 🐮 AgriSense AI - Guía de Uso

## Descripción

**AgriSense AI** es un sistema completo de detección y conteo de animales de granja en tiempo real, optimizado para ejecutarse en el **DJI Matrice 4TD** con Edge AI (10 TOPS). El proyecto implementa YOLOv8 con modificaciones específicas de DJI para inferencia en el hardware del dron.

## 📋 Características Principales

- ✅ **Detección en Tiempo Real**: Inferencia local en el chip del dron (baja latencia)
- ✅ **Visión Multispectral**: Soporte para cámara térmica (640x512) y RGB
- ✅ **Tracking Inteligente**: Evita contar el mismo animal múltiples veces
- ✅ **Hasta 10 Clases**: Vacas, ovejas, cerdos, caballos, etc.
- ✅ **Integración Cloud**: Envío automático de datos a DJI FlightHub 2
- ✅ **Análisis Térmico**: Monitoreo de salud basado en temperatura

## 🏗️ Estructura del Proyecto

```
CountingCows/
├── src/                        # Código fuente principal
│   ├── __init__.py
│   ├── detector.py            # Módulo de detección YOLOv8
│   ├── counter.py             # Sistema de conteo y tracking
│   └── cloud_api.py           # Cliente DJI FlightHub 2 API
├── scripts/                    # Scripts de utilidad
│   ├── train_model.py         # Script para entrenar modelos
│   └── demo.py                # Demo del sistema completo
├── configs/                    # Archivos de configuración
│   └── config.yaml            # Configuración principal
├── datasets/                   # Datasets de entrenamiento
├── models/                     # Modelos entrenados
├── results/                    # Resultados de procesamiento
├── main.py                     # Punto de entrada principal
├── requirements.txt            # Dependencias Python
└── README.md                   # Este archivo
```

## 🚀 Instalación

### 1. Clonar el repositorio

```bash
git clone <repository-url>
cd CountingCows
```

### 2. Crear entorno virtual (recomendado)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar MMYOLO (para entrenamiento)

```bash
python scripts/train_model.py --setup
```

Sigue las instrucciones impresas para:
1. Clonar MMYOLO
2. Cambiar al tag v0.6.0
3. Aplicar el patch DJI
4. Instalar dependencias

## 📊 Preparación del Dataset

### 1. Crear estructura de directorios

```bash
python scripts/train_model.py --prepare-dataset
```

### 2. Formato de Anotaciones

El dataset debe estar en formato **COCO JSON**:

```
datasets/coco/
├── annotations/
│   ├── instances_train2017.json
│   ├── instances_val2017.json
│   └── instances_test2017.json
├── train2017/
│   ├── 000000000001.jpg
│   └── ...
└── val2017/
    └── ...
```

### 3. Clases Soportadas (Máximo 10)

```python
0: "vaca"
1: "oveja"
2: "cerdo"
3: "caballo"
4: "cabra"
5: "gallina"
6: "pato"
7: "perro"
8: "gato"
9: "conejo"
```

**IMPORTANTE**: 
- Usar imágenes capturadas desde **perspectiva aérea**
- Incluir variedad de condiciones (luz, clima, alturas)
- Mínimo **500-1000 imágenes** para cuantificación óptima

## 🎓 Entrenamiento del Modelo

### 1. Generar archivo de configuración

```bash
python scripts/train_model.py --generate-config --num-classes 3
```

Esto crea `configs/yolov8_dji_custom.py` con las modificaciones DJI.

### 2. Entrenar el modelo

**Single GPU:**
```bash
cd mmyolo
python tools/train.py ../configs/yolov8_dji_custom.py
```

**Multi-GPU (4 GPUs):**
```bash
cd mmyolo
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./tools/dist_train.sh ../configs/yolov8_dji_custom.py 4
```

### 3. Validar el modelo

```bash
python tools/test.py ../configs/yolov8_dji_custom.py \
    work_dirs/yolov8_dji_custom/best_coco_bbox_mAP_epoch_X.pth \
    --show-dir ../results/
```

## 🔧 Cuantificación para DJI Edge AI

**El modelo .pth NO puede usarse directamente en el dron.** Debe ser cuantificado por DJI:

1. **Subir al Panel de DJI Developer**:
   - Ir a `Model Management`
   - Subir el archivo `.pth` entrenado
   - Subir **500-1000 imágenes** de calibración (JPG/PNG comprimidas)

2. **Proceso de Cuantificación**:
   - DJI procesará el modelo (puede tardar horas)
   - El modelo se optimiza para el chip de 10 TOPS del M4TD

3. **Descargar e Instalar**:
   - Descargar modelo cuantificado desde el panel
   - Instalar en **DJI Pilot 2** mediante *side-load*

## 💻 Uso del Sistema

### Configuración

Editar `configs/config.yaml` con tus parámetros:

```yaml
model:
  path: "models/yolov8_dji_quantized.pth"
  confidence_threshold: 0.5

dji_cloud:
  app_id: "YOUR_APP_ID"
  app_key: "YOUR_APP_KEY"
  workspace_id: "YOUR_WORKSPACE_ID"
```

### Procesar una Imagen

```bash
python main.py --image path/to/image.jpg --gps 40.4168 -3.7038 --altitude 50
```

### Procesar un Video

```bash
python main.py --video path/to/video.mp4 --report
```

### Ejecutar Demo

```bash
# Demo completo
python scripts/demo.py --full

# Solo detección y conteo
python scripts/demo.py --detection

# Solo integración Cloud
python scripts/demo.py --cloud

# Solo análisis térmico
python scripts/demo.py --thermal
```

## 📡 Integración con DJI FlightHub 2

### Obtener Credenciales

1. Registrarse en [DJI Developer Portal](https://developer.dji.com/)
2. Crear una aplicación
3. Obtener `app_id`, `app_key`, y `app_license`
4. Configurar workspace en FlightHub 2

### Envío Automático de Alertas

Cuando está habilitado en `config.yaml`, el sistema envía automáticamente:

- **Alertas de Detección**: Ubicación GPS, clase, conteo, confianza
- **Resúmenes de Misión**: Conteo total, área cubierta, estadísticas
- **Imágenes**: Imágenes con detecciones visualizadas
- **Datos Térmicos**: Análisis de temperatura (si está habilitado)

## 🔬 Análisis Térmico

Para usar la cámara térmica del M4TD:

```yaml
camera:
  use_thermal: true
  thermal_resolution: [640, 512]

thermal_analysis:
  enable_health_check: true
  temperature_thresholds:
    normal_min: 36.0
    normal_max: 39.0
    fever_threshold: 40.0
```

**Aplicaciones**:
- Detección nocturna de ganado
- Monitoreo de salud (fiebre, inflamación)
- Identificación de estrés térmico

## 📊 Salidas del Sistema

### Reportes JSON

```json
{
  "timestamp": "2025-11-26T10:30:00",
  "statistics": {
    "total_animals": 15,
    "by_class": {
      "vaca": 10,
      "oveja": 5
    },
    "active_tracks": 15
  },
  "tracked_animals": [...]
}
```

### Imágenes Procesadas

Guardadas en `results/` con bounding boxes y etiquetas.

### Mapas de Densidad

Visualización de concentración de animales por región.

## 🎯 Workflow en Producción

1. **Pre-vuelo**:
   - Cargar modelo cuantificado en DJI Pilot 2
   - Configurar misión en FlightHub 2
   - Verificar conectividad Cloud

2. **Durante el Vuelo**:
   - Inferencia en tiempo real en el Edge AI (10 TOPS)
   - Conteo automático con tracking
   - Envío de alertas a FlightHub 2

3. **Post-vuelo**:
   - Revisión de reportes y estadísticas
   - Análisis de mapas de densidad
   - Exportación de datos para gestión

## ⚠️ Limitaciones Importantes

- **Máximo 10 clases** (restricción de hardware DJI)
- **Input size fijo**: 640x640 píxeles
- **Modelo obligatorio**: `yolov8_s_syncbn_fast_8xb16-500e_coco.py`
- **Cuantificación obligatoria**: No usar .pth directamente
- **Perspectiva aérea**: Dataset debe ser capturado desde drones

## 🐛 Troubleshooting

### Error: "Modelo no cargado"
- Verificar que el modelo esté cuantificado por DJI
- Verificar ruta en `config.yaml`

### Error: "Autenticación DJI falló"
- Verificar credenciales en `config.yaml`
- Verificar conectividad a internet
- Verificar workspace_id

### Detecciones pobres
- Verificar que el dataset sea desde perspectiva aérea
- Aumentar tamaño del dataset
- Ajustar `confidence_threshold`

## 📚 Referencias

- [DJI Edge AI Documentation](https://developer.dji.com/)
- [MMYOLO GitHub](https://github.com/open-mmlab/mmyolo)
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [DJI FlightHub 2 API](https://developer.dji.com/flighthub-2)

## 📄 Licencia

Copyright © 2025 Arán Tecnologías

## 🤝 Soporte

Para preguntas o soporte técnico, contactar a Arán Tecnologías.

---

**Desarrollado para DJI Matrice 4TD con Edge AI (10 TOPS)**
