"""
Conftest for airflow dags/utils tests.

Adds the utils/ directory to sys.path so bare imports like
`from technical_indicators_calculator import ...` resolve when pytest
is invoked from the project root.

Resolves F-06-012 (audit 2026-04).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
