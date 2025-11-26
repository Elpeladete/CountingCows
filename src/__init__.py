"""
AgriSense AI: Detección y Conteo de Ganado en Tiempo Real
Sistema de detección optimizado para DJI Matrice 4TD con Edge AI
"""

__version__ = "1.0.0"
__author__ = "Arán Tecnologías"

from .detector import AnimalDetector
from .counter import AnimalCounter
from .cloud_api import DJICloudAPI

__all__ = ["AnimalDetector", "AnimalCounter", "DJICloudAPI"]
