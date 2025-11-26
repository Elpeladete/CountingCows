"""
Script de configuración inicial para AgriSense AI
Crea los directorios necesarios y configura el entorno
"""

import os
import sys
from pathlib import Path

def create_directories():
    """Crea la estructura de directorios necesaria"""
    dirs = [
        "logs",
        "results",
        "models",
        "datasets/coco/annotations",
        "datasets/coco/train2017",
        "datasets/coco/val2017",
        "datasets/coco/test2017",
        "datasets/coco/images/thermal"
    ]
    
    print("Creando estructura de directorios...")
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {dir_path}")
    
    print("\n✅ Estructura de directorios creada")

def check_config():
    """Verifica si existe el archivo de configuración"""
    config_path = Path("configs/config.yaml")
    
    if config_path.exists():
        print(f"\n✅ Archivo de configuración encontrado: {config_path}")
        print("\n⚠️  IMPORTANTE: Edita configs/config.yaml con tus credenciales DJI:")
        print("   - app_id")
        print("   - app_key")
        print("   - app_license")
        print("   - workspace_id")
    else:
        print(f"\n❌ No se encontró configs/config.yaml")

def print_next_steps():
    """Imprime los próximos pasos a seguir"""
    print("\n" + "=" * 70)
    print("PRÓXIMOS PASOS")
    print("=" * 70)
    print("""
1. CONFIGURACIÓN:
   - Editar configs/config.yaml con tus credenciales DJI
   - Ajustar parámetros del modelo y contador según necesidad

2. EJECUTAR DEMO (sin modelo real):
   python scripts/demo.py --full

3. PREPARAR DATASET:
   python scripts/train_model.py --prepare-dataset
   - Coloca tus imágenes en datasets/coco/train2017/
   - Coloca anotaciones en datasets/coco/annotations/

4. CONFIGURAR MMYOLO (para entrenamiento):
   python scripts/train_model.py --setup
   - Sigue las instrucciones para instalar MMYOLO v0.6.0

5. ENTRENAR MODELO:
   python scripts/train_model.py --generate-config --num-classes 3
   - Luego entrenar con MMYOLO según instrucciones

6. USAR EL SISTEMA:
   python main.py --image tu_imagen.jpg --gps 40.4168 -3.7038
   python main.py --video tu_video.mp4 --report
    """)
    print("=" * 70)

def main():
    print("=" * 70)
    print("CONFIGURACIÓN INICIAL - AgriSense AI")
    print("=" * 70)
    print()
    
    # Crear directorios
    create_directories()
    
    # Verificar configuración
    check_config()
    
    # Mostrar próximos pasos
    print_next_steps()

if __name__ == '__main__':
    main()
