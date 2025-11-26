#!/usr/bin/env python3
"""
Script principal para ejecutar el sistema AgriSense AI
Punto de entrada unificado para detección y conteo en producción
"""

import sys
import os
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2
import yaml
import argparse
import logging
from datetime import datetime
from typing import Optional

from src.detector import AnimalDetector
from src.counter import AnimalCounter
from src.cloud_api import DJICloudAPI, DetectionAlert

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AgriSenseAI:
    """Sistema principal de AgriSense AI"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Inicializa el sistema completo
        
        Args:
            config_path: Ruta al archivo de configuración
        """
        self.config = self._load_config(config_path)
        
        # Inicializar componentes
        self.detector = self._init_detector()
        self.counter = self._init_counter()
        self.cloud_api = self._init_cloud_api()
        
        logger.info("✅ AgriSense AI inicializado correctamente")
    
    def _load_config(self, config_path: str) -> dict:
        """Carga la configuración desde YAML"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuración cargada desde {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error al cargar configuración: {e}")
            raise
    
    def _init_detector(self) -> AnimalDetector:
        """Inicializa el detector"""
        model_config = self.config['model']
        camera_config = self.config['camera']
        
        return AnimalDetector(
            model_path=model_config['path'],
            conf_threshold=model_config['confidence_threshold'],
            nms_threshold=model_config['nms_threshold'],
            input_size=tuple(model_config['input_size']),
            use_thermal=camera_config['use_thermal']
        )
    
    def _init_counter(self) -> AnimalCounter:
        """Inicializa el contador"""
        counter_config = self.config['counter']
        
        return AnimalCounter(
            max_distance=counter_config['max_distance'],
            max_frames_missing=counter_config['max_frames_missing'],
            min_confidence=counter_config['min_confidence'],
            enable_tracking=counter_config['enable_tracking']
        )
    
    def _init_cloud_api(self) -> Optional[DJICloudAPI]:
        """Inicializa el cliente de Cloud API"""
        cloud_config = self.config['dji_cloud']
        
        if not cloud_config.get('auto_send_alerts', False):
            logger.info("Envío automático de alertas deshabilitado")
            return None
        
        return DJICloudAPI(
            api_base_url=cloud_config['api_base_url'],
            app_id=cloud_config['app_id'],
            app_key=cloud_config['app_key'],
            app_license=cloud_config['app_license'],
            workspace_id=cloud_config['workspace_id'],
            timeout=cloud_config['timeout']
        )
    
    def process_image(
        self,
        image_path: str,
        gps_coords: Optional[tuple] = None,
        altitude: float = None
    ):
        """
        Procesa una imagen individual
        
        Args:
            image_path: Ruta a la imagen
            gps_coords: Coordenadas GPS (lat, lon)
            altitude: Altitud en metros
        """
        # Cargar imagen
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"No se pudo cargar la imagen: {image_path}")
            return
        
        # Detectar animales
        timestamp = datetime.now().timestamp()
        detections = self.detector.detect(image, gps_coords, timestamp)
        
        # Actualizar contador
        counts = self.counter.update(detections, timestamp)
        
        logger.info(f"Procesada {image_path} - Detecciones: {len(detections)}, Conteo: {counts}")
        
        # Visualizar si está habilitado
        if self.config['output']['save_images']:
            viz_image = self.detector.visualize_detections(image, detections)
            
            output_dir = Path(self.config['output']['results_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = output_dir / f"result_{Path(image_path).stem}.jpg"
            cv2.imwrite(str(output_path), viz_image)
            logger.info(f"Imagen guardada en {output_path}")
    
    def process_video(self, video_path: str):
        """
        Procesa un video completo
        
        Args:
            video_path: Ruta al video
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"No se pudo abrir el video: {video_path}")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Procesando video: {video_path}")
        logger.info(f"FPS: {fps}, Total frames: {frame_count}")
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Procesar cada N frames (según FPS configurado)
            if frame_idx % max(1, int(fps / self.config['camera']['fps'])) == 0:
                timestamp = frame_idx / fps
                detections = self.detector.detect(frame, timestamp=timestamp)
                counts = self.counter.update(detections, timestamp)
                
                if frame_idx % 100 == 0:
                    logger.info(f"Frame {frame_idx}/{frame_count} - Conteo: {counts}")
            
            frame_idx += 1
        
        cap.release()
        
        # Mostrar estadísticas finales
        stats = self.counter.get_statistics()
        logger.info("=" * 70)
        logger.info("PROCESAMIENTO DE VIDEO COMPLETADO")
        logger.info(f"Total animales: {stats['total_animals']}")
        logger.info(f"Por clase: {stats['by_class']}")
        logger.info("=" * 70)
    
    def generate_report(self):
        """Genera y guarda un reporte del conteo"""
        output_dir = Path(self.config['output']['results_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"report_{timestamp}.json"
        
        report = self.counter.export_count_report(str(report_path))
        logger.info(f"📊 Reporte generado: {report_path}")
        
        return report


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description="AgriSense AI - Sistema de Detección y Conteo de Ganado"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Ruta al archivo de configuración'
    )
    parser.add_argument(
        '--image',
        type=str,
        help='Procesar una imagen individual'
    )
    parser.add_argument(
        '--video',
        type=str,
        help='Procesar un video'
    )
    parser.add_argument(
        '--gps',
        type=float,
        nargs=2,
        metavar=('LAT', 'LON'),
        help='Coordenadas GPS (latitud longitud)'
    )
    parser.add_argument(
        '--altitude',
        type=float,
        help='Altitud en metros'
    )
    parser.add_argument(
        '--report',
        action='store_true',
        help='Generar reporte al finalizar'
    )
    
    args = parser.parse_args()
    
    try:
        # Inicializar sistema
        system = AgriSenseAI(args.config)
        
        # Procesar según modo
        if args.image:
            gps_coords = tuple(args.gps) if args.gps else None
            system.process_image(args.image, gps_coords, args.altitude)
        
        elif args.video:
            system.process_video(args.video)
        
        else:
            parser.print_help()
            return
        
        # Generar reporte si se solicita
        if args.report:
            system.generate_report()
        
        logger.info("✅ Procesamiento completado exitosamente")
        
    except Exception as e:
        logger.error(f"❌ Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
