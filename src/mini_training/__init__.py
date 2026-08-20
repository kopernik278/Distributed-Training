"""Mini distributed training engine."""

from .config import TrainingConfig
from .parallel_state import initialize_model_parallel

__all__ = ["TrainingConfig", "initialize_model_parallel"]
