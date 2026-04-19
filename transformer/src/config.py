import random
import numpy as np
from typing import Any
from pathlib import Path
import yaml
from dataclasses import dataclass
import torch

from .logger import get_logger

logger = get_logger(__name__)

def load_config(cfg_path: Path) -> dict[str, Any]:
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    logger.info('Config loaded')
    return cfg

def set_seed(seed: int = 42, strict: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if strict:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f'Seed set to {seed} (strict={strict})')

@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int
    p: float
    pe: str
    max_len: int
    n_encoder: int
    n_decoder: int
    num_heads: int
    d_ff: int
