"""
Script para entrenar el modelo YOLOv8 con MMYOLO para DJI Edge AI
Sigue el flujo de trabajo especificado en el README
"""

import os
import sys
import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_mmyolo_environment():
    """
    Configura el entorno MMYOLO según especificaciones DJI
    """
    logger.info("=" * 70)
    logger.info("CONFIGURACIÓN DE MMYOLO PARA DJI EDGE AI")
    logger.info("=" * 70)
    
     instructions = """
     PASOS PARA ADAPTAR MMYOLO PARA DJI EDGE AI:
    
     1. Clona el repositorio MMYOLO:
         git clone https://github.com/open-mmlab/mmyolo.git
         cd mmyolo
    
     2. Cambia al tag v0.6.0 (OBLIGATORIO):
         git checkout tags/v0.6.0 -b dji-edge-ai
    
     3. Aplica el parche de compatibilidad DJI:
         git apply ../0001-NEW-ai-inside-init.patch
         # El parche adapta el modelo para el chip Edge AI del Matrice 4TD
    
     4. Instala dependencias:
         pip install -U openmim
         mim install "mmengine>=0.6.0"
         mim install "mmcv>=2.0.0rc4,<2.1.0"
         mim install "mmdet>=3.0.0,<4.0.0"
         pip install -v -e .
    
     5. Verifica instalación:
         python -c "import mmyolo; print(mmyolo.__version__)"
    
     6. IMPORTANTE: Usa siempre el archivo de configuración
         yolov8_s_syncbn_fast_8xb16-500e_coco.py como base.
         Limita el número de clases a 10 (por hardware DJI).
         Usa imágenes aéreas y térmicas (640x512) para el dataset.
     """
    
    print(instructions)
    logger.info("Entorno configurado. Proceda con la preparación del dataset.")


def prepare_dataset_structure():
    """
    Crea la estructura de directorios para el dataset en formato COCO
    """
    logger.info("=" * 70)
    logger.info("PREPARACIÓN DE ESTRUCTURA DE DATASET")
    logger.info("=" * 70)
    
    base_path = Path("../datasets/coco")
    
    dirs_to_create = [
        base_path / "annotations",
        base_path / "train2017",
        base_path / "val2017",
        base_path / "test2017",
        base_path / "images" / "thermal",  # Para imágenes térmicas
    ]
    
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Creado: {dir_path}")
    
    logger.info("\n" + "=" * 70)
    logger.info("FORMATO DE ANOTACIONES (COCO JSON)")
    logger.info("=" * 70)
    
    coco_format = """
    El dataset debe estar en formato COCO:
    
    datasets/coco/
    ├── annotations/
    │   ├── instances_train2017.json
    │   ├── instances_val2017.json
    │   └── instances_test2017.json
    ├── train2017/
    │   ├── 000000000001.jpg
    │   ├── 000000000002.jpg
    │   └── ...
    └── val2017/
        └── ...
    
    CLASES SOPORTADAS (máximo 10):
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
    
    IMPORTANTE:
    - Las imágenes deben ser capturadas desde perspectiva AÉREA
    - Incluir imágenes térmicas (640x512) si se usa cámara térmica
    - Mínimo 500-1000 imágenes para buena cuantificación
    - Anotar todos los animales visibles en cada imagen
    """
    
    print(coco_format)


def generate_training_config(num_classes: int = 3, data_root: str = "../datasets/coco"):
    """
    Genera el archivo de configuración para entrenamiento
    
    Args:
        num_classes: Número de clases (máximo 10)
        data_root: Ruta al directorio de datos
    """
    if num_classes > 10:
        logger.error("❌ ERROR: El número máximo de clases es 10 (limitación DJI)")
        sys.exit(1)
    
    logger.info("=" * 70)
    logger.info(f"GENERANDO CONFIGURACIÓN PARA {num_classes} CLASES")
    logger.info("=" * 70)
    
    config_template = f"""# Configuración YOLOv8 para DJI Edge AI
# Basado en: configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py
# Modificado según parche DJI

_base_ = ['../_base_/default_runtime.py', '../_base_/det_p5_tta.py']

# ========================Parámetros Frecuentemente Modificados======================
# -----Datos-----
data_root = '{data_root}/'
train_ann_file = 'annotations/instances_train2017.json'
train_data_prefix = 'train2017/'
val_ann_file = 'annotations/instances_val2017.json'
val_data_prefix = 'val2017/'

# -----Modelo-----
num_classes = {num_classes}  # MÁXIMO 10 clases para DJI Edge AI
img_scale = (640, 640)  # Tamaño de entrada (obligatorio para DJI)

# -----Entrenamiento-----
max_epochs = 500
base_lr = 0.01
train_batch_size_per_gpu = 16
train_num_workers = 8

# ========================Configuración del Modelo======================
model = dict(
    type='YOLODetector',
    data_preprocessor=dict(
        type='YOLOv5DetDataPreprocessor',
        mean=[128., 128., 128.],  # Modificado según patch DJI
        std=[128., 128., 128.],   # Modificado según patch DJI
        bgr_to_rgb=True
    ),
    backbone=dict(
        type='YOLOv8CSPDarknet',
        deepen_factor=0.33,
        widen_factor=0.5,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='ReLU', inplace=True)  # SiLU → ReLU (DJI)
    ),
    neck=dict(
        type='YOLOv8PAFPN',
        deepen_factor=0.33,
        widen_factor=0.5,
        in_channels=[256, 512, 512],
        out_channels=[256, 512, 512],
        num_csp_blocks=3,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='ReLU', inplace=True)  # SiLU → ReLU (DJI)
    ),
    bbox_head=dict(
        type='YOLOv8Head',
        head_module=dict(
            type='YOLOv8HeadModule',
            num_classes=num_classes,
            in_channels=[256, 512, 512],
            widen_factor=0.5,
            reg_max=16,
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='ReLU', inplace=True),  # SiLU → ReLU (DJI)
            featmap_strides=[8, 16, 32],
            skip_dfl=False  # Añadido por patch DJI
        )
    )
)

# ========================Dataset y DataLoader======================
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix)
    )
)

val_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root=data_root,
        ann_file=val_ann_file,
        data_prefix=dict(img=val_data_prefix)
    )
)

# ========================Optimización======================
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=base_lr, momentum=0.937, weight_decay=0.0005),
    constructor='YOLOv5OptimizerConstructor'
)

# ========================Hooks y Runtime======================
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        max_keep_ckpts=3,
        save_best='auto'
    ),
    logger=dict(type='LoggerHook', interval=50)
)
"""
    
    config_path = Path("../configs/yolov8_dji_custom.py")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_template)
    
    logger.info(f"✓ Configuración guardada en: {config_path}")
    logger.info(f"✓ Número de clases: {num_classes}")
    logger.info(f"✓ Data root: {data_root}")


def print_training_instructions():
    """
    Imprime las instrucciones para entrenar el modelo
    """
    logger.info("\n" + "=" * 70)
    logger.info("COMANDOS PARA ENTRENAR EL MODELO")
    logger.info("=" * 70)
    
    instructions = """
    ENTRENAMIENTO SINGLE GPU:
    -------------------------
    python tools/train.py configs/yolov8/yolov8_dji_custom.py
    
    ENTRENAMIENTO MULTI-GPU (4 GPUs):
    ----------------------------------
    bash ./tools/dist_train.sh configs/yolov8/yolov8_dji_custom.py 4
    
    O usando torchrun:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
        --nproc_per_node=4 \
        --master_port=29500 \
        tools/train.py configs/yolov8/yolov8_dji_custom.py \
        --launcher pytorch
    
    VALIDACIÓN:
    -----------
    python tools/test.py configs/yolov8/yolov8_dji_custom.py \
        work_dirs/yolov8_dji_custom/epoch_500.pth \
        --show-dir results/
    
    EXPORTAR A ONNX (para testing local):
    --------------------------------------
    python tools/deployment/pytorch2onnx.py \
        configs/yolov8/yolov8_dji_custom.py \
        work_dirs/yolov8_dji_custom/best_coco_bbox_mAP_epoch_X.pth \
        --output-file yolov8_dji.onnx \
        --input-img demo/demo.jpg \
        --shape 640 640
    
    FLUJO DE CUANTIFICACIÓN Y DESPLIEGUE DJI:
    -----------------------------------------
    1. Una vez entrenado, sube el archivo .pth al panel de DJI Developer (Model Management).
    2. Sube 500-1000 imágenes de calibración (JPG/PNG, aéreas y térmicas si aplica).
    3. DJI procesará la cuantificación (puede tardar varias horas).
    4. Descarga el modelo cuantificado optimizado para Matrice 4TD.
    5. Instala el modelo en DJI Pilot 2 usando side-load.
    6. El modelo final ejecutará en el chip Edge AI (10 TOPS) del dron.
    
    ADVERTENCIAS:
    - Máximo 10 clases (por hardware DJI)
    - Input size fijo: 640x640
    - Solo usar yolov8_s_syncbn_fast_8xb16-500e_coco.py como base
    - Dataset debe ser capturado desde drones (perspectiva aérea)
    - Imágenes térmicas deben ser 640x512 px
    """
    
    print(instructions)


def main():
    """Función principal del script"""
    parser = argparse.ArgumentParser(
        description="Script de entrenamiento para AgriSense AI - DJI Edge AI"
    )
    parser.add_argument(
        '--setup',
        action='store_true',
        help='Mostrar instrucciones de configuración de MMYOLO'
    )
    parser.add_argument(
        '--prepare-dataset',
        action='store_true',
        help='Crear estructura de directorios para dataset'
    )
    parser.add_argument(
        '--generate-config',
        action='store_true',
        help='Generar archivo de configuración de entrenamiento'
    )
    parser.add_argument(
        '--num-classes',
        type=int,
        default=3,
        help='Número de clases a detectar (máximo 10, default: 3 [vaca, oveja, cerdo])'
    )
    parser.add_argument(
        '--data-root',
        type=str,
        default='../datasets/coco',
        help='Ruta al directorio de datos (default: ../datasets/coco)'
    )
    parser.add_argument(
        '--show-instructions',
        action='store_true',
        help='Mostrar instrucciones de entrenamiento'
    )
    
    args = parser.parse_args()
    
    # Si no se pasa ningún argumento, mostrar help
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    # Ejecutar según opciones
    if args.setup:
        setup_mmyolo_environment()
    
    if args.prepare_dataset:
        prepare_dataset_structure()
    
    if args.generate_config:
        generate_training_config(args.num_classes, args.data_root)
    
    if args.show_instructions:
        print_training_instructions()
    
    logger.info("\n✅ Script completado exitosamente")


if __name__ == '__main__':
    main()
