"""
Módulo de conteo y tracking de animales
Implementa algoritmos para contar animales evitando duplicados
"""

import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, deque
from dataclasses import dataclass, field
import logging
from datetime import datetime

from .detector import Detection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrackedAnimal:
    """Representa un animal trackeado a través de múltiples frames"""
    track_id: int
    class_name: str
    positions: deque = field(default_factory=lambda: deque(maxlen=30))  # Últimas 30 posiciones
    confidences: deque = field(default_factory=lambda: deque(maxlen=30))
    first_seen: float = 0.0
    last_seen: float = 0.0
    frames_missing: int = 0
    is_counted: bool = False
    gps_coords: Optional[Tuple[float, float]] = None
    
    def update(self, detection: Detection, timestamp: float):
        """Actualiza el tracking con una nueva detección"""
        self.positions.append(detection.center)
        self.confidences.append(detection.confidence)
        self.last_seen = timestamp
        self.frames_missing = 0
        
        if detection.gps_coords:
            self.gps_coords = detection.gps_coords
    
    def get_predicted_position(self) -> Tuple[float, float]:
        """Predice la siguiente posición basada en el movimiento"""
        if len(self.positions) < 2:
            return self.positions[-1] if self.positions else (0, 0)
        
        # Predicción lineal simple
        p1 = np.array(self.positions[-2])
        p2 = np.array(self.positions[-1])
        velocity = p2 - p1
        predicted = p2 + velocity
        
        return tuple(predicted)
    
    def get_avg_confidence(self) -> float:
        """Retorna la confianza promedio"""
        return np.mean(self.confidences) if self.confidences else 0.0


class AnimalCounter:
    """
    Sistema de conteo de animales con tracking
    Evita contar el mismo animal múltiples veces
    """
    
    def __init__(
        self,
        max_distance: float = 50.0,
        max_frames_missing: int = 30,
        min_confidence: float = 0.5,
        enable_tracking: bool = True
    ):
        """
        Inicializa el contador de animales
        
        Args:
            max_distance: Distancia máxima (píxeles) para asociar detecciones
            max_frames_missing: Frames máximos antes de considerar perdido un track
            min_confidence: Confianza mínima para contar un animal
            enable_tracking: Si False, cuenta sin tracking (puede tener duplicados)
        """
        self.max_distance = max_distance
        self.max_frames_missing = max_frames_missing
        self.min_confidence = min_confidence
        self.enable_tracking = enable_tracking
        
        # Estado del tracking
        self.active_tracks: Dict[int, TrackedAnimal] = {}
        self.next_track_id = 0
        self.counted_animals: Dict[str, int] = defaultdict(int)
        self.total_count = 0
        
        # Historial
        self.detection_history: List[List[Detection]] = []
        self.count_history: List[Dict[str, int]] = []
        
        logger.info(f"Contador inicializado - Tracking: {enable_tracking}")
        logger.info(f"Max distancia: {max_distance}px, Max frames perdidos: {max_frames_missing}")
    
    def _calculate_iou(self, bbox1: Tuple[int, int, int, int], 
                       bbox2: Tuple[int, int, int, int]) -> float:
        """Calcula Intersection over Union entre dos bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calcular intersección
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calcular áreas
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_distance(self, pos1: Tuple[float, float], 
                           pos2: Tuple[float, float]) -> float:
        """Calcula distancia euclidiana entre dos posiciones"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def _match_detections_to_tracks(
        self,
        detections: List[Detection],
        timestamp: float
    ) -> Tuple[Dict[int, Detection], List[Detection]]:
        """
        Asocia detecciones con tracks existentes
        
        Returns:
            Tupla de (matched_tracks, unmatched_detections)
        """
        matched_tracks = {}
        unmatched_detections = []
        
        if not self.active_tracks:
            return matched_tracks, detections
        
        # Matriz de costos (distancia entre cada detección y cada track)
        cost_matrix = np.zeros((len(detections), len(self.active_tracks)))
        
        track_ids = list(self.active_tracks.keys())
        
        for i, det in enumerate(detections):
            for j, track_id in enumerate(track_ids):
                track = self.active_tracks[track_id]
                
                # Solo asociar si es la misma clase
                if det.class_name != track.class_name:
                    cost_matrix[i, j] = float('inf')
                    continue
                
                # Usar posición predicha si está disponible
                if track.positions:
                    track_pos = track.get_predicted_position()
                    distance = self._calculate_distance(det.center, track_pos)
                    cost_matrix[i, j] = distance
                else:
                    cost_matrix[i, j] = float('inf')
        
        # Asignación simple (greedy)
        used_detections = set()
        used_tracks = set()
        
        while True:
            min_cost = float('inf')
            min_i, min_j = -1, -1
            
            for i in range(len(detections)):
                if i in used_detections:
                    continue
                for j in range(len(track_ids)):
                    if j in used_tracks:
                        continue
                    if cost_matrix[i, j] < min_cost and cost_matrix[i, j] < self.max_distance:
                        min_cost = cost_matrix[i, j]
                        min_i, min_j = i, j
            
            if min_i == -1:
                break
            
            matched_tracks[track_ids[min_j]] = detections[min_i]
            used_detections.add(min_i)
            used_tracks.add(min_j)
        
        # Detecciones no asociadas
        for i, det in enumerate(detections):
            if i not in used_detections:
                unmatched_detections.append(det)
        
        return matched_tracks, unmatched_detections
    
    def update(self, detections: List[Detection], timestamp: float = None) -> Dict[str, int]:
        """
        Actualiza el contador con nuevas detecciones
        
        Args:
            detections: Lista de detecciones del frame actual
            timestamp: Timestamp del frame (se genera automáticamente si no se provee)
            
        Returns:
            Diccionario con conteo actual por clase
        """
        if timestamp is None:
            timestamp = datetime.now().timestamp()
        
        # Guardar en historial
        self.detection_history.append(detections)
        
        if not self.enable_tracking:
            # Conteo simple sin tracking
            current_counts = defaultdict(int)
            for det in detections:
                if det.confidence >= self.min_confidence:
                    current_counts[det.class_name] += 1
            
            self.count_history.append(dict(current_counts))
            return dict(current_counts)
        
        # Tracking habilitado
        # 1. Asociar detecciones con tracks existentes
        matched_tracks, unmatched_detections = self._match_detections_to_tracks(
            detections, timestamp
        )
        
        # 2. Actualizar tracks asociados
        for track_id, detection in matched_tracks.items():
            self.active_tracks[track_id].update(detection, timestamp)
        
        # 3. Crear nuevos tracks para detecciones no asociadas
        for detection in unmatched_detections:
            if detection.confidence >= self.min_confidence:
                track = TrackedAnimal(
                    track_id=self.next_track_id,
                    class_name=detection.class_name,
                    first_seen=timestamp,
                    last_seen=timestamp,
                    gps_coords=detection.gps_coords
                )
                track.update(detection, timestamp)
                
                # Contar el nuevo animal
                if not track.is_counted:
                    self.counted_animals[detection.class_name] += 1
                    self.total_count += 1
                    track.is_counted = True
                    logger.info(f"🐄 Nuevo {detection.class_name} detectado (ID: {self.next_track_id})")
                
                self.active_tracks[self.next_track_id] = track
                self.next_track_id += 1
        
        # 4. Incrementar frames_missing para tracks no asociados
        tracks_to_remove = []
        for track_id, track in self.active_tracks.items():
            if track_id not in matched_tracks:
                track.frames_missing += 1
                
                # Eliminar tracks perdidos
                if track.frames_missing > self.max_frames_missing:
                    tracks_to_remove.append(track_id)
                    logger.debug(f"Track {track_id} ({track.class_name}) eliminado - no visto por {track.frames_missing} frames")
        
        for track_id in tracks_to_remove:
            del self.active_tracks[track_id]
        
        # Guardar conteo actual en historial
        self.count_history.append(dict(self.counted_animals))
        
        return dict(self.counted_animals)
    
    def get_current_count(self) -> Dict[str, int]:
        """Retorna el conteo actual de animales"""
        return dict(self.counted_animals)
    
    def get_total_count(self) -> int:
        """Retorna el conteo total de todos los animales"""
        return self.total_count
    
    def get_active_tracks_count(self) -> int:
        """Retorna el número de tracks activos actualmente"""
        return len(self.active_tracks)
    
    def reset(self):
        """Reinicia el contador y todos los tracks"""
        self.active_tracks.clear()
        self.counted_animals.clear()
        self.total_count = 0
        self.next_track_id = 0
        self.detection_history.clear()
        self.count_history.clear()
        logger.info("Contador reiniciado")
    
    def get_statistics(self) -> Dict:
        """
        Retorna estadísticas completas del conteo
        """
        stats = {
            "total_animals": self.total_count,
            "by_class": dict(self.counted_animals),
            "active_tracks": self.get_active_tracks_count(),
            "frames_processed": len(self.detection_history),
            "average_confidence": 0.0
        }
        
        # Calcular confianza promedio de tracks activos
        if self.active_tracks:
            avg_confidences = [track.get_avg_confidence() 
                             for track in self.active_tracks.values()]
            stats["average_confidence"] = float(np.mean(avg_confidences))
        
        return stats
    
    def export_count_report(self, output_path: str = None) -> Dict:
        """
        Genera un reporte completo del conteo para enviar a FlightHub 2
        
        Args:
            output_path: Ruta opcional para guardar el reporte en JSON
            
        Returns:
            Diccionario con reporte completo
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "statistics": self.get_statistics(),
            "tracked_animals": []
        }
        
        # Agregar información de cada animal trackeado
        for track_id, track in self.active_tracks.items():
            animal_info = {
                "track_id": track_id,
                "class": track.class_name,
                "confidence": track.get_avg_confidence(),
                "first_seen": datetime.fromtimestamp(track.first_seen).isoformat(),
                "last_seen": datetime.fromtimestamp(track.last_seen).isoformat(),
                "gps_coords": track.gps_coords,
                "position_history": [list(pos) for pos in track.positions]
            }
            report["tracked_animals"].append(animal_info)
        
        if output_path:
            import json
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"Reporte exportado a {output_path}")
        
        return report
    
    def get_density_map(self, image_shape: Tuple[int, int], 
                       grid_size: int = 50) -> np.ndarray:
        """
        Genera un mapa de densidad de animales detectados
        Útil para visualización de concentración de ganado
        
        Args:
            image_shape: (height, width) de la imagen
            grid_size: Tamaño de cada celda del grid en píxeles
            
        Returns:
            Matriz 2D con conteo de animales por región
        """
        height, width = image_shape
        grid_h = height // grid_size
        grid_w = width // grid_size
        
        density_map = np.zeros((grid_h, grid_w), dtype=np.int32)
        
        for track in self.active_tracks.values():
            if track.positions:
                cx, cy = track.positions[-1]
                grid_x = min(int(cx // grid_size), grid_w - 1)
                grid_y = min(int(cy // grid_size), grid_h - 1)
                density_map[grid_y, grid_x] += 1
        
        return density_map
