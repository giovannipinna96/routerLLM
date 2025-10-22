"""
RouterLLM: Intelligent LLM Router for optimizing accuracy/cost efficiency
"""

import warnings
# Suppress the pynvml deprecation warning from PyTorch CUDA initialization
warnings.filterwarnings("ignore", message="The pynvml package is deprecated")

__version__ = "0.1.0"
__author__ = "RouterLLM Team"

from .models.llm_manager import LLMManager
from .models.router import BERTRouter, DummyRouter
from .utils.carbon_tracker import CarbonTracker
from .utils.logger import setup_logger

__all__ = [
    "LLMManager",
    "BERTRouter",
    "DummyRouter",
    "CarbonTracker",
    "setup_logger"
]