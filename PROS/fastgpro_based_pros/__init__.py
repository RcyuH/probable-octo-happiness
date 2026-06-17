"""FastGRPO-based PROS migration package.

The package intentionally avoids importing torch-heavy training modules at
import time. Use ``train_pros.py`` or ``ProsTrainer`` for actual training.
"""

from .pros_config import ProsConfig

__all__ = ["ProsConfig"]
