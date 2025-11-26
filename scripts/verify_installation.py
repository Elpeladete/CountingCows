"""
Script para verificar que todas las dependencias están correctamente instaladas
"""

import sys
from typing import List, Tuple

def check_package(package_name: str, import_name: str = None) -> Tuple[bool, str]:
    """
    Verifica si un paquete está instalado e importable
    
    Args:
        package_name: Nombre del paquete
        import_name: Nombre para importar (si es diferente)
    
    Returns:
        Tupla (success, version_or_error)
    """
    if import_name is None:
        import_name = package_name
    
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        return True, version
    except ImportError as e:
        return False, str(e)

def main():
    """Función principal de verificación"""
    print("=" * 70)
    print("VERIFICACIÓN DE INSTALACIÓN - AgriSense AI")
    print("=" * 70)
    print(f"\nPython Version: {sys.version}\n")
    
    # Lista de paquetes a verificar
    packages = [
        ("numpy", "numpy"),
        ("opencv-python", "cv2"),
        ("PyYAML", "yaml"),
        ("Pillow", "PIL"),
        ("scipy", "scipy"),
        ("matplotlib", "matplotlib"),
        ("requests", "requests"),
        ("loguru", "loguru"),
        ("pytest", "pytest"),
    ]
    
    print("Verificando paquetes instalados:\n")
    
    all_ok = True
    results = []
    
    for package_name, import_name in packages:
        success, info = check_package(package_name, import_name)
        results.append((package_name, success, info))
        
        status = "✅" if success else "❌"
        print(f"{status} {package_name:20s} - ", end="")
        
        if success:
            print(f"v{info}")
        else:
            print(f"ERROR: {info}")
            all_ok = False
    
    print("\n" + "=" * 70)
    
    if all_ok:
        print("✅ TODAS LAS DEPENDENCIAS BÁSICAS INSTALADAS CORRECTAMENTE")
        print("\nPróximos pasos:")
        print("1. Para entrenamiento, instalar MMYOLO:")
        print("   python scripts/train_model.py --setup")
        print("\n2. Ejecutar el demo:")
        print("   python scripts/demo.py --full")
        print("\n3. Configurar credenciales en configs/config.yaml")
    else:
        print("❌ ALGUNAS DEPENDENCIAS FALTAN O TIENEN ERRORES")
        print("\nInstalar dependencias faltantes con:")
        print("pip install -r requirements.txt")
    
    print("=" * 70)
    
    return 0 if all_ok else 1

if __name__ == '__main__':
    sys.exit(main())
