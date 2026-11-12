"""mini-chatgpt: a small ChatGPT-style model (pretrain -> SFT -> API) built for game NPCs."""

from .config import ModelConfig, TrainConfig, load_config
from .model import GPT

__all__ = ["ModelConfig", "TrainConfig", "load_config", "GPT"]
