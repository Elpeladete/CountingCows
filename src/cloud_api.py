"""
Cliente para integración con DJI FlightHub 2 Cloud API
Envía alertas de detección, ubicación GPS y conteo a plataformas de terceros
"""

import requests
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DetectionAlert:
    """Alerta de detección para enviar a FlightHub 2"""
    timestamp: str
    drone_id: str
    mission_id: str
    animal_class: str
    count: int
    gps_latitude: float
    gps_longitude: float
    altitude: float
    confidence: float
    thermal_data: Optional[Dict] = None
    image_url: Optional[str] = None


class DJICloudAPI:
    """
    Cliente para DJI FlightHub 2 Cloud API
    Permite enviar datos de detección a sistemas de gestión de terceros
    """
    
    def __init__(
        self,
        api_base_url: str,
        app_id: str,
        app_key: str,
        app_license: str,
        workspace_id: str,
        timeout: int = 30
    ):
        """
        Inicializa el cliente de DJI Cloud API
        
        Args:
            api_base_url: URL base de la API de FlightHub 2
            app_id: ID de la aplicación registrada en DJI Developer
            app_key: Clave de la aplicación
            app_license: Licencia de la aplicación
            workspace_id: ID del workspace en FlightHub 2
            timeout: Timeout para requests HTTP (segundos)
        """
        self.api_base_url = api_base_url.rstrip('/')
        self.app_id = app_id
        self.app_key = app_key
        self.app_license = app_license
        self.workspace_id = workspace_id
        self.timeout = timeout
        
        self.session = requests.Session()
        self.access_token = None
        self.token_expires_at = None
        
        logger.info(f"DJI Cloud API inicializado - Workspace: {workspace_id}")
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """
        Genera los headers de autenticación para las requests
        
        Returns:
            Diccionario con headers de autenticación
        """
        headers = {
            "Content-Type": "application/json",
            "X-App-Id": self.app_id,
            "X-App-Key": self.app_key,
            "X-App-License": self.app_license,
        }
        
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        
        return headers
    
    def authenticate(self) -> bool:
        """
        Autentica con la API de DJI FlightHub 2
        
        Returns:
            True si la autenticación fue exitosa
        """
        try:
            endpoint = f"{self.api_base_url}/auth/token"
            
            payload = {
                "app_id": self.app_id,
                "app_key": self.app_key,
                "app_license": self.app_license,
                "workspace_id": self.workspace_id
            }
            
            response = self.session.post(
                endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                self.access_token = data.get("access_token")
                expires_in = data.get("expires_in", 3600)
                
                from datetime import timedelta
                self.token_expires_at = datetime.now() + timedelta(seconds=expires_in)
                
                logger.info("✅ Autenticación exitosa con DJI FlightHub 2")
                return True
            else:
                logger.error(f"❌ Error de autenticación: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error al autenticar: {e}")
            return False
    
    def is_token_valid(self) -> bool:
        """Verifica si el token de acceso es válido"""
        if not self.access_token or not self.token_expires_at:
            return False
        
        return datetime.now() < self.token_expires_at
    
    def send_detection_alert(self, alert: DetectionAlert) -> bool:
        """
        Envía una alerta de detección a FlightHub 2
        
        Args:
            alert: Objeto DetectionAlert con la información
            
        Returns:
            True si el envío fue exitoso
        """
        # Verificar autenticación
        if not self.is_token_valid():
            logger.info("Token expirado, re-autenticando...")
            if not self.authenticate():
                return False
        
        try:
            endpoint = f"{self.api_base_url}/workspaces/{self.workspace_id}/events"
            
            # Convertir alerta a diccionario
            payload = asdict(alert)
            
            response = self.session.post(
                endpoint,
                json=payload,
                headers=self._get_auth_headers(),
                timeout=self.timeout
            )
            
            if response.status_code in [200, 201]:
                logger.info(f"✅ Alerta enviada: {alert.animal_class} x{alert.count} en ({alert.gps_latitude}, {alert.gps_longitude})")
                return True
            else:
                logger.error(f"❌ Error al enviar alerta: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error al enviar alerta: {e}")
            return False
    
    def send_batch_alerts(self, alerts: List[DetectionAlert]) -> Dict[str, int]:
        """
        Envía múltiples alertas en lote
        
        Args:
            alerts: Lista de alertas a enviar
            
        Returns:
            Diccionario con estadísticas de envío {'success': N, 'failed': M}
        """
        stats = {"success": 0, "failed": 0}
        
        for alert in alerts:
            if self.send_detection_alert(alert):
                stats["success"] += 1
            else:
                stats["failed"] += 1
        
        logger.info(f"📊 Batch enviado - Exitosos: {stats['success']}, Fallidos: {stats['failed']}")
        return stats
    
    def send_count_summary(
        self,
        drone_id: str,
        mission_id: str,
        count_by_class: Dict[str, int],
        gps_coords: tuple,
        altitude: float,
        area_covered: float = None
    ) -> bool:
        """
        Envía un resumen de conteo al finalizar un vuelo o sección
        
        Args:
            drone_id: ID del dron
            mission_id: ID de la misión
            count_by_class: Diccionario con conteo por clase
            gps_coords: (latitud, longitud) del punto central
            altitude: Altitud promedio del vuelo
            area_covered: Área cubierta en hectáreas (opcional)
            
        Returns:
            True si el envío fue exitoso
        """
        if not self.is_token_valid():
            if not self.authenticate():
                return False
        
        try:
            endpoint = f"{self.api_base_url}/workspaces/{self.workspace_id}/mission-reports"
            
            payload = {
                "timestamp": datetime.now().isoformat(),
                "drone_id": drone_id,
                "mission_id": mission_id,
                "report_type": "animal_count_summary",
                "location": {
                    "latitude": gps_coords[0],
                    "longitude": gps_coords[1],
                    "altitude": altitude
                },
                "count_data": count_by_class,
                "total_count": sum(count_by_class.values()),
                "area_covered_hectares": area_covered
            }
            
            response = self.session.post(
                endpoint,
                json=payload,
                headers=self._get_auth_headers(),
                timeout=self.timeout
            )
            
            if response.status_code in [200, 201]:
                logger.info(f"✅ Resumen de conteo enviado - Total: {sum(count_by_class.values())} animales")
                return True
            else:
                logger.error(f"❌ Error al enviar resumen: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error al enviar resumen: {e}")
            return False
    
    def upload_detection_image(
        self,
        image_path: str,
        detection_metadata: Dict
    ) -> Optional[str]:
        """
        Sube una imagen con detecciones a FlightHub 2
        
        Args:
            image_path: Ruta local de la imagen
            detection_metadata: Metadata de la detección
            
        Returns:
            URL de la imagen subida, o None si falla
        """
        if not self.is_token_valid():
            if not self.authenticate():
                return None
        
        try:
            endpoint = f"{self.api_base_url}/workspaces/{self.workspace_id}/media/upload"
            
            with open(image_path, 'rb') as f:
                files = {'file': f}
                data = {'metadata': json.dumps(detection_metadata)}
                
                headers = self._get_auth_headers()
                headers.pop("Content-Type")  # Dejar que requests maneje multipart
                
                response = self.session.post(
                    endpoint,
                    files=files,
                    data=data,
                    headers=headers,
                    timeout=self.timeout * 2  # Más tiempo para uploads
                )
            
            if response.status_code in [200, 201]:
                data = response.json()
                image_url = data.get("url")
                logger.info(f"✅ Imagen subida: {image_url}")
                return image_url
            else:
                logger.error(f"❌ Error al subir imagen: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error al subir imagen: {e}")
            return None
    
    def get_mission_status(self, mission_id: str) -> Optional[Dict]:
        """
        Obtiene el estado actual de una misión
        
        Args:
            mission_id: ID de la misión
            
        Returns:
            Diccionario con información de la misión, o None si falla
        """
        if not self.is_token_valid():
            if not self.authenticate():
                return None
        
        try:
            endpoint = f"{self.api_base_url}/workspaces/{self.workspace_id}/missions/{mission_id}"
            
            response = self.session.get(
                endpoint,
                headers=self._get_auth_headers(),
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"❌ Error al obtener estado de misión: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error al obtener estado: {e}")
            return None
    
    def send_health_check(self) -> bool:
        """
        Envía un health check al sistema
        Útil para verificar conectividad durante el vuelo
        
        Returns:
            True si el sistema está operativo
        """
        if not self.is_token_valid():
            if not self.authenticate():
                return False
        
        try:
            endpoint = f"{self.api_base_url}/health"
            
            response = self.session.get(
                endpoint,
                headers=self._get_auth_headers(),
                timeout=5
            )
            
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"❌ Health check falló: {e}")
            return False
