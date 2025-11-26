"""
Módulo de detección de animales usando YOLOv8 optimizado para DJI Edge AI
Soporta detección en imágenes RGB y térmicas
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Clase para almacenar información de una detección"""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    center: Tuple[float, float]  # (cx, cy)
    area: float
    timestamp: float
    gps_coords: Optional[Tuple[float, float]] = None  # (lat, lon)


class AnimalDetector:
    """
    Detector de animales optimizado para DJI Matrice 4TD
    Soporta hasta 10 clases de animales según limitaciones de hardware
    """
    
    # Clases soportadas (máximo 10 según restricción DJI)
    ANIMAL_CLASSES = {
        0: "vaca",
        1: "oveja",
        2: "cerdo",
        3: "caballo",
        4: "cabra",
        5: "gallina",
        6: "pato",
        7: "perro",
        8: "gato",
        9: "conejo"
    }
    
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        input_size: Tuple[int, int] = (640, 640),
        use_thermal: bool = False
    ):
        """
        Inicializa el detector de animales
        
        Args:
            model_path: Ruta al modelo cuantificado (.pth o .onnx)
            conf_threshold: Umbral de confianza para detecciones
            nms_threshold: Umbral para Non-Maximum Suppression
            input_size: Tamaño de entrada del modelo (debe ser 640x640 para DJI)
            use_thermal: Si True, procesa imágenes térmicas (640x512 del M4TD)
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.use_thermal = use_thermal
        
        self.model = None
        self.is_initialized = False
        
        logger.info(f"Detector inicializado - Modelo: {model_path}")
        logger.info(f"Modo térmico: {use_thermal}, Umbral confianza: {conf_threshold}")
        
    def load_model(self):
        """
        Carga el modelo YOLOv8 cuantificado para DJI
        En producción, este modelo se carga directamente en el chip de 10 TOPS del M4TD
        """
        try:
            # TODO: Implementar carga del modelo según formato (.pth, .onnx, o formato DJI)
            # En el Edge AI del dron, esto se maneja automáticamente por DJI Pilot 2
            logger.info("⚠️ Modelo no cargado - Implementar carga según formato DJI")
            logger.info("En producción, el modelo se ejecuta directamente en el Edge AI del M4TD")
            self.is_initialized = True
            
        except Exception as e:
            logger.error(f"Error al cargar el modelo: {e}")
            raise
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocesa la imagen según configuración del modelo YOLOv8
        Normalización: mean=[128, 128, 128], std=[128, 128, 128]
        
        Args:
            image: Imagen BGR de OpenCV o imagen térmica
            
        Returns:
            Tensor preprocesado listo para inferencia
        """
        # Redimensionar a input_size
        resized = cv2.resize(image, self.input_size)
        
        # Para imágenes térmicas, convertir a RGB si es necesario
        if self.use_thermal and len(resized.shape) == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        
        # Convertir BGR a RGB
        if not self.use_thermal:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalización según patch DJI: mean=128, std=128
        normalized = (resized - 128.0) / 128.0
        
        # Convertir a formato tensor (1, 3, H, W)
        tensor = np.transpose(normalized, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0).astype(np.float32)
        
        return tensor
    
    def postprocess_detections(
        self,
        outputs: np.ndarray,
        original_shape: Tuple[int, int]
    ) -> List[Detection]:
        """
        Post-procesa las salidas del modelo YOLOv8
        
        Args:
            outputs: Salidas brutas del modelo
            original_shape: (height, width) de la imagen original
            
        Returns:
            Lista de detecciones filtradas y procesadas
        """
        detections = []
        
        # TODO: Implementar post-procesamiento según formato de salida YOLOv8
        # Incluye: decodificación de bboxes, aplicación de NMS, filtrado por confianza
        
        logger.info(f"Post-procesamiento completado: {len(detections)} detecciones")
        return detections
    
    def detect(
        self,
        image: np.ndarray,
        gps_coords: Optional[Tuple[float, float]] = None,
        timestamp: Optional[float] = None
    ) -> List[Detection]:
        """
        Detecta animales en una imagen
        
        Args:
            image: Imagen BGR (RGB) o térmica (Grayscale)
            gps_coords: Coordenadas GPS opcionales (lat, lon)
            timestamp: Timestamp de la captura
            
        Returns:
            Lista de detecciones con información completa
        """
        if not self.is_initialized:
            logger.warning("Modelo no inicializado. Llamando a load_model()...")
            self.load_model()
        
        original_shape = image.shape[:2]
        
        # Preprocesar imagen
        input_tensor = self.preprocess_image(image)
        
        # Inferencia
        # TODO: Ejecutar inferencia con el modelo
        # outputs = self.model(input_tensor)
        
        # Post-procesar detecciones
        # detections = self.postprocess_detections(outputs, original_shape)
        
        # Por ahora, retornamos lista vacía hasta implementar la inferencia real
        detections = []
        
        # Agregar metadata a cada detección
        for det in detections:
            det.gps_coords = gps_coords
            det.timestamp = timestamp if timestamp else 0.0
        
        logger.info(f"Detectados {len(detections)} animales en la imagen")
        return detections
    
    def detect_batch(
        self,
        images: List[np.ndarray],
        gps_coords_list: Optional[List[Tuple[float, float]]] = None,
        timestamps: Optional[List[float]] = None
    ) -> List[List[Detection]]:
        """
        Detecta animales en un lote de imágenes (optimizado para throughput)
        
        Args:
            images: Lista de imágenes
            gps_coords_list: Lista de coordenadas GPS
            timestamps: Lista de timestamps
            
        Returns:
            Lista de listas de detecciones
        """
        results = []
        
        for idx, image in enumerate(images):
            gps = gps_coords_list[idx] if gps_coords_list else None
            ts = timestamps[idx] if timestamps else None
            
            detections = self.detect(image, gps, ts)
            results.append(detections)
        
        return results
    
    def visualize_detections(
        self,
        image: np.ndarray,
        detections: List[Detection],
        show_labels: bool = True,
        color_map: Optional[Dict[str, Tuple[int, int, int]]] = None
    ) -> np.ndarray:
        """
        Visualiza las detecciones en la imagen
        
        Args:
            image: Imagen original
            detections: Lista de detecciones
            show_labels: Si True, muestra etiquetas con clase y confianza
            color_map: Mapa de colores por clase
            
        Returns:
            Imagen con detecciones dibujadas
        """
        viz_image = image.copy()
        
        default_color = (0, 255, 0)
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = default_color
            
            if color_map and det.class_name in color_map:
                color = color_map[det.class_name]
            
            # Dibujar bbox
            cv2.rectangle(viz_image, (x1, y1), (x2, y2), color, 2)
            
            # Dibujar centro
            cx, cy = int(det.center[0]), int(det.center[1])
            cv2.circle(viz_image, (cx, cy), 4, color, -1)
            
            if show_labels:
                label = f"{det.class_name}: {det.confidence:.2f}"
                
                # Fondo para el texto
                (text_w, text_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    viz_image,
                    (x1, y1 - text_h - 4),
                    (x1 + text_w, y1),
                    color,
                    -1
                )
                
                # Texto
                cv2.putText(
                    viz_image,
                    label,
                    (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
        
        return viz_image
    
    def get_thermal_stats(self, thermal_image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict:
        """
        Calcula estadísticas de temperatura en una región térmica
        Útil para análisis de salud del ganado
        
        Args:
            thermal_image: Imagen térmica (640x512 del M4TD)
            bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            Diccionario con estadísticas térmicas
        """
        x1, y1, x2, y2 = bbox
        roi = thermal_image[y1:y2, x1:x2]
        
        stats = {
            "temp_mean": float(np.mean(roi)),
            "temp_max": float(np.max(roi)),
            "temp_min": float(np.min(roi)),
            "temp_std": float(np.std(roi)),
            "area_pixels": int(roi.size)
        }
        
        return stats
