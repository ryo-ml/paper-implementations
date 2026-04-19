from typing import Any
from pathlib import Path
import datasets
import torch
import torch.nn as nn

from .config import ModelConfig
from .data_loader import _preprocess_dataset
from .tokenizer import AbstractTokenizer, BytePairEncoding, WordTokenizer
from .model import SinusoidalPositionalEncoding, Transformer
from .logger import get_logger


BASE_DIR = Path(__file__).resolve().parent.parent
TOKENIZER_DIR = BASE_DIR / 'tokenizers'
CKPT_DIR = BASE_DIR / 'checkpoints'

logger = get_logger(__name__)

def build_tokenizer(cfg: dict[str, Any]) -> AbstractTokenizer:
    TOKENIZER_REGISTRY = {
        'bpe': BytePairEncoding,
        'word': WordTokenizer,
    }

    tokenizer_cfg = cfg['tokenizer']

    path = TOKENIZER_DIR / tokenizer_cfg['fname']
    tokenizer_class = TOKENIZER_REGISTRY[tokenizer_cfg['type']]

    if path is None:
        raise ValueError('path is required for tokenizer save/load')

    if path.exists():
        tokenizer = tokenizer_class.load(path)
        logger.info(f'Tokenizer loaded: {tokenizer_cfg}')
    else:
        tokenizer = tokenizer_class(vocab_size=cfg['model']['vocab_size'])
        dataset = datasets.load_dataset(cfg['dataset']['name'])
        dataset = _preprocess_dataset(dataset)
        texts = list(dataset['train']['source']) + list(dataset['train']['target'])

        tokenizer.train(texts)
        tokenizer.save(path)
        logger.info(f'Tokenizer trained and saved: {tokenizer_cfg}')

    return tokenizer

def build_model(
    cfg: dict[str, Any],
    pad_token_id: int = 0,
    bos_token_id: int = 2,
    eos_token_id: int = 3,
) -> nn.Module:
    '''
    Builds model with the passed config. 
    '''
    device = cfg['device']
    model_cfg = ModelConfig(**cfg['model'])

    pe_type = model_cfg.pe
    max_len = model_cfg.max_len
    d_model = model_cfg.d_model

    if pe_type.lower() == 'sinusoidal':
        pe = SinusoidalPositionalEncoding(max_len, d_model)
    else:
        raise ValueError(f"Unsupported PE type: {pe_type}")
    
    model = Transformer(
        vocab_size=model_cfg.vocab_size,
        d_model=model_cfg.d_model,
        p=model_cfg.p,
        pe=pe,
        max_len=model_cfg.max_len,
        n_encoder=model_cfg.n_encoder,
        n_decoder=model_cfg.n_decoder,
        num_heads=model_cfg.num_heads,
        d_ff=model_cfg.d_ff,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
    )

    logger.info(f"Model built: {model_cfg}, pad_token_id: {pad_token_id}")
    return model.to(device)

def build_optimizer(
    cfg: dict[str, Any],
    model: nn.Module
) -> torch.optim.Optimizer:
    '''
    Builds optimizer with the passed config. 
    '''
    optim_kwargs = cfg['optimizer'].copy()
    optim_type = optim_kwargs.pop('type')

    if 'scheduler' in optim_kwargs:
        optim_kwargs.pop('scheduler')

    optim_class = getattr(torch.optim, optim_type, None)

    if optim_class is None:
        raise ValueError(f"Unsupported Optimizer type: {optim_type}")

    optim = optim_class(
        model.parameters(),
        **optim_kwargs,
    )

    logger.info(f"Optimizer built: {optim_type} {optim_kwargs}")
    return optim

def build_scheduler(
    cfg: dict[str, Any],
    optim: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.LRScheduler:
    '''
    Builds the scheduler. Follows the original implementation from `Attention is All You Need.`

    Corresponds to increasing the learning rate linearly for the first `warmup_steps` training steps,
    and decreasing it thereafter proportionally to the inverse square root of the step number.
    '''
    scheduler_type = cfg['optimizer']['scheduler']['type']

    if scheduler_type.lower() == 'constant':
        scheduler = torch.optim.lr_scheduler.ConstantLR(optim, 1)

    elif scheduler_type.lower() == 'lambda':
        assert cfg['optimizer']['lr'] == 1.0, 'lr must be 1.0 for the lambda scheduler'

        d_model = cfg['model']['d_model']
        warmup_steps = cfg['optimizer']['scheduler']['warmup_steps']

        def _lambda(step: int) -> float:
            step = max(1, step)

            lr = d_model ** -0.5 * min(step ** -0.5, step * warmup_steps ** -1.5)
            return lr

        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=_lambda)

    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_type}")

    logger.info(f"Scheduler built: {scheduler}")
    return scheduler

def build_loss_fn(
    cfg: dict[str, Any],
    pad_token_id: int = 0
) -> nn.Module:
    '''
    Builds loss function with the passed config.
    '''
    loss_type = cfg['train']['loss_fn']

    if loss_type.lower() == 'cross_entropy':
        loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    else:
        raise ValueError(f"Unsupported loss_fn type: {loss_type}")
    logger.info(f"Loss set to {loss_fn}")    

    return loss_fn

def save_checkpoint(
    cfg: dict[str, Any],
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    loss: list[list[float]],
) -> None:
    '''
    Saves checkpoint to `path`. 
    '''
    path = CKPT_DIR / cfg['train']['checkpoint']['fname']

    if path.exists():
        raise FileExistsError(f"Checkpoint already exists at {path}.")

    path.parent.mkdir(parents=True, exist_ok=True)

    d = {
        'model': model.state_dict(),
        'optim': optim.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'loss': loss,
    }

    torch.save(d, path)
    logger.info(f"Checkpoint saved to {path}")

def load_checkpoint(
    cfg: dict[str, Any],
    model: torch.nn.Module,
    optim: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> tuple[torch.nn.Module, torch.optim.Optimizer | None, torch.optim.lr_scheduler.LRScheduler | None, int, list[float]]:
    '''
    Loads checkpoint from `path`.
    '''
    path = CKPT_DIR / cfg['train']['checkpoint']['fname']

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {path}")

    d = torch.load(path)

    model.load_state_dict(d['model'])
    if optim is not None:
        optim.load_state_dict(d['optim'])
    if scheduler is not None:
        scheduler.load_state_dict(d['scheduler'])
    epoch = d['epoch']
    loss = d['loss']
    logger.info(f"Checkpoint loaded from {path}")

    return model, optim, scheduler, epoch, loss
