"""
Script de demostración para probar el algoritmo de detección y conteo
Simula el flujo de trabajo completo desde detección hasta envío a FlightHub 2
"""

import sys
import os
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np
import argparse
import logging
from datetime import datetime

from detector import AnimalDetector, Detection
from counter import AnimalCounter
from cloud_api import DJICloudAPI, DetectionAlert

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_mock_detection(
    class_id: int,
    class_name: str,
    bbox: tuple,
    confidence: float,
    gps_coords: tuple = None
) -> Detection:
    """Crea una detección simulada para testing"""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    area = (x2 - x1) * (y2 - y1)
    
    return Detection(
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        bbox=bbox,
        center=(cx, cy),
        area=area,
        timestamp=datetime.now().timestamp(),
        gps_coords=gps_coords
    )


def demo_detection_and_counting():
    """Demuestra el sistema de detección y conteo"""
    logger.info("=" * 70)
    logger.info("DEMO: DETECCIÓN Y CONTEO DE ANIMALES")
    logger.info("=" * 70)
    
    # Inicializar detector
    detector = AnimalDetector(
        model_path="models/yolov8_dji_quantized.pth",
        conf_threshold=0.5,
        use_thermal=False
    )
    
    # Inicializar contador con tracking
    counter = AnimalCounter(
        max_distance=50.0,
        max_frames_missing=30,
        enable_tracking=True
    )
    
    # Simular secuencia de frames con detecciones
    logger.info("\n📹 Simulando secuencia de video...")
    
    # Frame 1: 3 vacas detectadas
    frame1_detections = [
        create_mock_detection(0, "vaca", (100, 100, 200, 200), 0.95, (40.4168, -3.7038)),
        create_mock_detection(0, "vaca", (300, 150, 400, 250), 0.92, (40.4169, -3.7039)),
        create_mock_detection(0, "vaca", (500, 200, 600, 300), 0.88, (40.4170, -3.7040)),
    ]
    
    counts = counter.update(frame1_detections, datetime.now().timestamp())
    logger.info(f"Frame 1 - Conteo: {counts}")
    
    # Frame 2: Las mismas vacas se mueven ligeramente + 2 ovejas nuevas
    frame2_detections = [
        create_mock_detection(0, "vaca", (110, 105, 210, 205), 0.94, (40.4168, -3.7038)),
        create_mock_detection(0, "vaca", (310, 155, 410, 255), 0.91, (40.4169, -3.7039)),
        create_mock_detection(0, "vaca", (510, 205, 610, 305), 0.89, (40.4170, -3.7040)),
        create_mock_detection(1, "oveja", (700, 100, 750, 150), 0.87, (40.4171, -3.7041)),
        create_mock_detection(1, "oveja", (800, 120, 850, 170), 0.85, (40.4172, -3.7042)),
    ]
    
    counts = counter.update(frame2_detections, datetime.now().timestamp() + 0.1)
    logger.info(f"Frame 2 - Conteo: {counts}")
    
    # Frame 3: Continúa el tracking
    frame3_detections = [
        create_mock_detection(0, "vaca", (120, 110, 220, 210), 0.93),
        create_mock_detection(0, "vaca", (320, 160, 420, 260), 0.90),
        create_mock_detection(0, "vaca", (520, 210, 620, 310), 0.87),
        create_mock_detection(1, "oveja", (710, 105, 760, 155), 0.86),
        create_mock_detection(1, "oveja", (810, 125, 860, 175), 0.84),
    ]
    
    counts = counter.update(frame3_detections, datetime.now().timestamp() + 0.2)
    logger.info(f"Frame 3 - Conteo: {counts}")
    
    # Mostrar estadísticas finales
    logger.info("\n" + "=" * 70)
    logger.info("ESTADÍSTICAS FINALES")
    logger.info("=" * 70)
    
    stats = counter.get_statistics()
    logger.info(f"Total de animales contados: {stats['total_animals']}")
    logger.info(f"Por clase: {stats['by_class']}")
    logger.info(f"Tracks activos: {stats['active_tracks']}")
    logger.info(f"Frames procesados: {stats['frames_processed']}")
    logger.info(f"Confianza promedio: {stats['average_confidence']:.2f}")
    
    # Exportar reporte
    report_path = "../results/count_report.json"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    report = counter.export_count_report(report_path)
    logger.info(f"\n📊 Reporte exportado a: {report_path}")
    
    return counter, stats


def demo_cloud_integration(counter: AnimalCounter, stats: dict):
    """Demuestra la integración con DJI FlightHub 2"""
    logger.info("\n" + "=" * 70)
    logger.info("DEMO: INTEGRACIÓN CON DJI FLIGHTHUB 2")
    logger.info("=" * 70)
    
    # Inicializar cliente de Cloud API (con credenciales de ejemplo)
    cloud_api = DJICloudAPI(
        api_base_url="https://api.dji-flighthub.com/v1",
        app_id="demo_app_id",
        app_key="demo_app_key",
        app_license="demo_license",
        workspace_id="demo_workspace"
    )
    
    logger.info("⚠️ NOTA: Usando credenciales de DEMO (no funcionales)")
    logger.info("Para producción, configure credenciales reales en config.yaml\n")
    
    # Crear alerta de detección de ejemplo
    alert = DetectionAlert(
        timestamp=datetime.now().isoformat(),
        drone_id="M4TD-001",
        mission_id="MISSION-2025-001",
        animal_class="vaca",
        count=3,
        gps_latitude=40.4168,
        gps_longitude=-3.7038,
        altitude=50.0,
        confidence=0.92,
        thermal_data={
            "temp_mean": 35.5,
            "temp_max": 37.2,
            "temp_min": 34.1
        }
    )
    
    logger.info(f"📡 Alerta de ejemplo creada:")
    logger.info(f"   - Clase: {alert.animal_class}")
    logger.info(f"   - Conteo: {alert.count}")
    logger.info(f"   - GPS: ({alert.gps_latitude}, {alert.gps_longitude})")
    logger.info(f"   - Altitud: {alert.altitude}m")
    logger.info(f"   - Confianza: {alert.confidence:.2f}")
    
    # Simular envío de resumen de conteo
    logger.info("\n📤 Enviando resumen de conteo a FlightHub 2...")
    
    success = cloud_api.send_count_summary(
        drone_id="M4TD-001",
        mission_id="MISSION-2025-001",
        count_by_class=stats['by_class'],
        gps_coords=(40.4168, -3.7038),
        altitude=50.0,
        area_covered=2.5  # hectáreas
    )
    
    if success:
        logger.info("✅ Resumen enviado exitosamente")
    else:
        logger.info("⚠️ No se pudo enviar (credenciales de demo)")


def demo_thermal_analysis():
    """Demuestra análisis de imágenes térmicas"""
    logger.info("\n" + "=" * 70)
    logger.info("DEMO: ANÁLISIS DE IMÁGENES TÉRMICAS")
    logger.info("=" * 70)
    
    # Crear imagen térmica simulada (640x512)
    thermal_image = np.random.randint(25, 40, (512, 640), dtype=np.uint8)
    
    # Simular una región "caliente" (animal)
    thermal_image[200:300, 300:400] = np.random.randint(35, 38, (100, 100))
    
    detector = AnimalDetector(
        model_path="models/yolov8_dji_thermal.pth",
        use_thermal=True
    )
    
    # Obtener estadísticas térmicas de una región
    bbox = (300, 200, 400, 300)
    thermal_stats = detector.get_thermal_stats(thermal_image, bbox)
    
    logger.info(f"📊 Estadísticas térmicas de región detectada:")
    logger.info(f"   - Temperatura media: {thermal_stats['temp_mean']:.1f}°C")
    logger.info(f"   - Temperatura máxima: {thermal_stats['temp_max']:.1f}°C")
    logger.info(f"   - Temperatura mínima: {thermal_stats['temp_min']:.1f}°C")
    logger.info(f"   - Desviación estándar: {thermal_stats['temp_std']:.2f}°C")
    logger.info(f"   - Área: {thermal_stats['area_pixels']} píxeles")
    
    logger.info("\n💡 Aplicaciones del análisis térmico:")
    logger.info("   - Detección nocturna de ganado")
    logger.info("   - Monitoreo de salud (fiebre, inflamación)")
    logger.info("   - Identificación de animales en estrés térmico")
    logger.info("   - Seguimiento de patrones de comportamiento")


def main():
    """Función principal del demo"""
    parser = argparse.ArgumentParser(
        description="Demo del sistema AgriSense AI - Detección y Conteo de Ganado"
    )
    parser.add_argument(
        '--full',
        action='store_true',
        help='Ejecutar demo completo (detección + conteo + cloud + térmico)'
    )
    parser.add_argument(
        '--detection',
        action='store_true',
        help='Demo de detección y conteo únicamente'
    )
    parser.add_argument(
        '--cloud',
        action='store_true',
        help='Demo de integración con Cloud API'
    )
    parser.add_argument(
        '--thermal',
        action='store_true',
        help='Demo de análisis térmico'
    )
    
    args = parser.parse_args()
    
    # Si no se pasa ningún argumento, ejecutar demo completo
    if not any([args.full, args.detection, args.cloud, args.thermal]):
        args.full = True
    
    try:
        if args.full or args.detection:
            counter, stats = demo_detection_and_counting()
        
        if args.full or args.cloud:
            if not (args.full or args.detection):
                # Crear datos simulados si solo se ejecuta cloud
                counter = AnimalCounter()
                stats = {'by_class': {'vaca': 3, 'oveja': 2}, 'total_animals': 5}
            demo_cloud_integration(counter, stats)
        
        if args.full or args.thermal:
            demo_thermal_analysis()
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ DEMO COMPLETADO EXITOSAMENTE")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"\n❌ Error durante el demo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
