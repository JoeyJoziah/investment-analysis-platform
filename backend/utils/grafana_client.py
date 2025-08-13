"""
Grafana API Client for Dashboard Management
"""

import os
import json
import requests
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class GrafanaClient:
    """Client for interacting with Grafana API"""
    
    def __init__(self):
        self.base_url = os.getenv('GRAFANA_URL', 'https://autodailynewsletterintake.grafana.net')
        self.api_key = os.getenv('GRAFANA_API_KEY')
        self.org_id = os.getenv('GRAFANA_ORG_ID', '1')
        
        if not self.api_key:
            logger.warning("Grafana API key not configured. Dashboard features will be limited.")
        
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'X-Grafana-Org-Id': self.org_id
        }
    
    def create_annotation(self, text: str, tags: List[str] = None, dashboard_id: int = None) -> bool:
        """Create an annotation in Grafana (e.g., for deployments, important events)"""
        if not self.api_key:
            return False
            
        try:
            data = {
                'text': text,
                'tags': tags or ['investment-app'],
                'time': int(datetime.now().timestamp() * 1000),
                'timeEnd': int(datetime.now().timestamp() * 1000)
            }
            
            if dashboard_id:
                data['dashboardId'] = dashboard_id
            
            response = requests.post(
                f'{self.base_url}/api/annotations',
                headers=self.headers,
                json=data
            )
            
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to create Grafana annotation: {e}")
            return False
    
    def create_alert(self, name: str, condition: Dict[str, Any], message: str) -> bool:
        """Create an alert rule in Grafana"""
        if not self.api_key:
            return False
            
        try:
            alert_data = {
                'uid': f'alert-{name.lower().replace(" ", "-")}',
                'title': name,
                'condition': condition,
                'data': [
                    {
                        'refId': 'A',
                        'queryType': '',
                        'model': condition
                    }
                ],
                'noDataState': 'NoData',
                'execErrState': 'Alerting',
                'for': '5m',
                'annotations': {
                    'description': message
                },
                'labels': {
                    'app': 'investment-analysis'
                }
            }
            
            response = requests.post(
                f'{self.base_url}/api/v1/provisioning/alert-rules',
                headers=self.headers,
                json=alert_data
            )
            
            return response.status_code in [200, 201]
        except Exception as e:
            logger.error(f"Failed to create Grafana alert: {e}")
            return False
    
    def get_dashboard_url(self, dashboard_uid: str) -> Optional[str]:
        """Get the URL for a specific dashboard"""
        if not dashboard_uid:
            return None
        return f"{self.base_url}/d/{dashboard_uid}"
    
    def test_connection(self) -> bool:
        """Test the connection to Grafana"""
        try:
            response = requests.get(
                f'{self.base_url}/api/health',
                headers=self.headers
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to connect to Grafana: {e}")
            return False
    
    def create_investment_dashboard(self) -> Optional[str]:
        """Create a custom investment analysis dashboard"""
        if not self.api_key:
            return None
            
        dashboard = {
            'dashboard': {
                'title': 'Investment Analysis Platform',
                'tags': ['investment', 'stocks', 'ml'],
                'timezone': 'browser',
                'panels': [
                    {
                        'id': 1,
                        'title': 'API Usage vs Limits',
                        'type': 'graph',
                        'gridPos': {'x': 0, 'y': 0, 'w': 12, 'h': 8},
                        'targets': [
                            {
                                'expr': 'api_calls_total',
                                'legendFormat': '{{provider}}'
                            }
                        ]
                    },
                    {
                        'id': 2,
                        'title': 'Model Predictions Accuracy',
                        'type': 'stat',
                        'gridPos': {'x': 12, 'y': 0, 'w': 6, 'h': 4},
                        'targets': [
                            {
                                'expr': 'model_accuracy_score'
                            }
                        ]
                    },
                    {
                        'id': 3,
                        'title': 'Daily Recommendations Generated',
                        'type': 'stat',
                        'gridPos': {'x': 18, 'y': 0, 'w': 6, 'h': 4},
                        'targets': [
                            {
                                'expr': 'recommendations_generated_total'
                            }
                        ]
                    },
                    {
                        'id': 4,
                        'title': 'Cost Tracking',
                        'type': 'gauge',
                        'gridPos': {'x': 12, 'y': 4, 'w': 12, 'h': 4},
                        'targets': [
                            {
                                'expr': 'monthly_cost_usd'
                            }
                        ],
                        'options': {
                            'max': 50,
                            'thresholds': {
                                'steps': [
                                    {'value': 0, 'color': 'green'},
                                    {'value': 40, 'color': 'yellow'},
                                    {'value': 45, 'color': 'red'}
                                ]
                            }
                        }
                    }
                ],
                'version': 1
            },
            'overwrite': True
        }
        
        try:
            response = requests.post(
                f'{self.base_url}/api/dashboards/db',
                headers=self.headers,
                json=dashboard
            )
            
            if response.status_code in [200, 201]:
                result = response.json()
                return result.get('url')
            return None
        except Exception as e:
            logger.error(f"Failed to create dashboard: {e}")
            return None


# Singleton instance
grafana_client = GrafanaClient()