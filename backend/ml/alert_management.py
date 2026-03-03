"""
Alert management for model monitoring.

Extracted from model_monitoring.py.  Contains the AlertManager class only.
Import via the original path (backend.ml.model_monitoring) or directly from here.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any

try:
    from backend.ml.monitoring_types import AlertSeverity, ModelAlert
except ImportError:  # pragma: no cover
    from monitoring_types import AlertSeverity, ModelAlert  # type: ignore[no-redef]

logger = logging.getLogger(__name__)


class AlertManager:
    """Manages model monitoring alerts"""

    def __init__(self, storage_path: str = "/app/monitoring/alerts") -> None:
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.alerts: List[ModelAlert] = []
        self.alert_handlers: Dict[str, Callable] = {}
        self.lock = threading.Lock()

        self._load_alerts()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def create_alert(
        self,
        model_name: str,
        alert_type: str,
        severity: AlertSeverity,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a new alert"""

        alert_id = (
            f"{model_name}_{alert_type}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        )

        alert = ModelAlert(
            id=alert_id,
            timestamp=datetime.now(timezone.utc),
            model_name=model_name,
            alert_type=alert_type,
            severity=severity,
            message=message,
            details=details or {},
        )

        with self.lock:
            self.alerts.append(alert)

        self._save_alert(alert)
        self._trigger_alert_handlers(alert)

        logger.warning(f"Alert created: {alert_id} - {message}")
        return alert_id

    def resolve_alert(self, alert_id: str, resolution_notes: str = "") -> None:
        """Resolve an alert"""

        with self.lock:
            for alert in self.alerts:
                if alert.id == alert_id:
                    alert.is_resolved = True
                    alert.resolved_at = datetime.now(timezone.utc)
                    alert.resolution_notes = resolution_notes
                    self._save_alert(alert)
                    logger.info(f"Alert resolved: {alert_id}")
                    break

    def get_active_alerts(self, model_name: Optional[str] = None) -> List[ModelAlert]:
        """Get active (unresolved) alerts"""

        active_alerts = [alert for alert in self.alerts if not alert.is_resolved]
        if model_name:
            active_alerts = [a for a in active_alerts if a.model_name == model_name]
        return active_alerts

    def register_alert_handler(self, alert_type: str, handler: Callable) -> None:
        """Register custom alert handler"""
        self.alert_handlers[alert_type] = handler

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _trigger_alert_handlers(self, alert: ModelAlert) -> None:
        """Trigger registered alert handlers"""

        if 'generic' in self.alert_handlers:
            try:
                self.alert_handlers['generic'](alert)
            except Exception as exc:
                logger.error(f"Error in generic alert handler: {exc}")

        if alert.alert_type in self.alert_handlers:
            try:
                self.alert_handlers[alert.alert_type](alert)
            except Exception as exc:
                logger.error(f"Error in {alert.alert_type} alert handler: {exc}")

    def _load_alerts(self) -> None:
        """Load alerts from storage"""
        try:
            alerts_file = self.storage_path / "alerts.json"
            if alerts_file.exists():
                with open(alerts_file, 'r') as fh:
                    alerts_data = json.load(fh)

                for alert_data in alerts_data:
                    alert_data['timestamp'] = datetime.fromisoformat(alert_data['timestamp'])
                    alert_data['severity'] = AlertSeverity(alert_data['severity'])
                    if alert_data.get('resolved_at'):
                        alert_data['resolved_at'] = datetime.fromisoformat(
                            alert_data['resolved_at']
                        )
                    self.alerts.append(ModelAlert(**alert_data))

                logger.info(f"Loaded {len(self.alerts)} alerts")
        except Exception as exc:
            logger.error(f"Error loading alerts: {exc}")

    def _save_alert(self, alert: ModelAlert) -> None:
        """Save individual alert"""
        try:
            alert_file = self.storage_path / f"alert_{alert.id}.json"
            with open(alert_file, 'w') as fh:
                json.dump(alert.to_dict(), fh, indent=2)
        except Exception as exc:
            logger.error(f"Error saving alert: {exc}")


__all__ = ["AlertManager"]
