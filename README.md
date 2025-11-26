# 🐮 AgriSense AI: Detección y Conteo de Ganado en Tiempo Real (Edge AI DJI Enterprise)

## Descripción del Proyecto

**AgriSense AI** es un proyecto de *Machine Learning* especializado en la detección y el conteo de animales de granja en tiempo real. Utiliza la plataforma de Inteligencia Artificial en el Borde (*Edge AI*) de DJI Enterprise para ejecutar la inferencia directamente en el *hardware* del dron (Matrice 4TD), eliminando la dependencia de la conectividad en campo y acelerando la toma de decisiones para la gestión de inventario y monitoreo pecuario.

La solución se centra inicialmente en el **monitoreo de vacas (Cattle)**, aprovechando la visión térmica para la detección nocturna y el análisis de salud, pero está diseñada para ser escalable a otras especies.

## Características Clave

*   **Detección en Tiempo Real:** Algoritmo optimizado para ejecutar la inferencia localmente en el chip de cómputo del dron (10 TOPS), garantizando baja latencia y conteo inmediato. [1]
*   **Visión Multispectral:** Utiliza la cámara térmica (640x512 px) del Matrice 4TD para el monitoreo eficaz en condiciones de baja visibilidad y detección de firmas de calor (salud o ubicación). [2]
*   **Monitoreo Escalable:** Arquitectura diseñada para la detección de hasta **10 categorías de objetos** distintas, permitiendo expandir fácilmente la funcionalidad a **cerdos, ovejas, caballos y más**. [3]
*   **Integración Cloud:** Uso de la **Cloud API** de DJI FlightHub 2 para enviar automáticamente alertas de detección (ubicación GPS, tiempo y conteo) a plataformas de gestión de terceros del cliente. [4, 5]
*   **Trazabilidad:** Las detecciones son registradas en los *logs* de misión, asegurando un rastro de auditoría completo. [6]

## Stack Tecnológico

| Componente | Tecnología | Rol en el Proyecto |
| :--- | :--- | :--- |
| **Hardware de Inferencia** | DJI Matrice 4TD / DJI Dock 3 | Ejecución del algoritmo en el Edge (10 TOPS) y captura de datos térmicos. [1] |
| **Framework ML** | MMYOLO (Basado en PyTorch) | Toolbox recomendado para el entrenamiento de modelos YOLOv8 de terceros. [3] |
| **Modelo Base** | YOLOv8 (yolov8_s_syncbn_fast...) | Arquitectura de detección de objetos ligeros, adaptada para el *Edge*. [3] |
| **Plataforma Cloud** | DJI FlightHub 2 Enterprise | Orquestación de misiones y puente de comunicación (OpenAPI) para el envío de datos. [7, 4] |
| **Datos** | *Dataset* de Ganado Aéreo (Térmico/RGB) | Datos específicos para evitar el fallo del modelo en perspectiva aérea (*domain shift*). [8] |

## Guía de Desarrollo y Entrenamiento

El entrenamiento requiere un flujo de trabajo estricto dictado por DJI para la **optimización (*quantization*)** en el hardware del dron.

### 1. Preparación del Código (Adaptación al Hardware DJI)

El modelo de código abierto debe ser modificado para cumplir con las especificaciones del chip del M4TD:

1.  **Clonar el Proyecto Base:** Descargar el *framework* MMYOLO desde GitHub. [3]
2.  **Establecer la Versión:** Cambiar el proyecto MMYOLO al *tag* **`v0.6.0`**. [3]
3.  **Aplicar el Parche:** Aplicar el archivo `.patch` descargado para inyectar la compatibilidad con DJI. [3]
4.  **Configurar Clases:** En el archivo de configuración, asegurarse de que la cantidad de categorías a detectar (**Vaca, Cerdo, Oveja, etc.**) **no exceda las 10 clases** (`num_classes <= 10`). [3]
5.  **Limitar el *Config File*:** Utilizar el archivo de configuración obligatorio: `yolov8_s_syncbn_fast_8xb16-500e_coco.py`. [3]

### 2. Entrenamiento y Validación Local

1.  **Entrenar el Modelo:** Utilizar el *dataset* de ganado (etiquetado) para entrenar el modelo adaptado.
    *   *Comando de ejemplo:* `CUDA_VISIBLE_DEVICES=0,1,2,3./tools/dist_train.sh configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py 4` [3]
2.  **Validar el `.pth`:** Asegurar que el archivo de peso resultante (`.pth`) cumpla con los requisitos de precisión antes del siguiente paso.

### 3. Despliegue (Cuantificación Obligatoria)

El modelo **no puede ser cargado directamente** al dron. Debe ser optimizado por DJI.

1.  **Subir los Archivos:** En el panel de desarrollador, en la sección `[Model Management]`, subir:
    *   El archivo de peso **`.pth`** entrenado localmente. [3]
    *   Imágenes de Calibración: Se recomienda subir entre **500 y 1,000 imágenes** (en formato JPG, JPEG, PNG o BMP comprimido) para el proceso de cuantificación. [3]
2.  **Iniciar la Cuantificación:** DJI procesará el modelo para optimizarlo (reducción de precisión, *quantization*) para el chip de la serie Matrice. [3]
3.  **Descarga y *Side Load*:** Una vez que DJI valide el modelo, el archivo final (ejecutable en el dron) estará disponible para su descarga e instalación en el DJI Pilot 2 (mediante *side load*). [9]
