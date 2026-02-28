import logging
from pathlib import Path
from typing import Dict, Any
import yaml
import random
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt

from .model_cifar import ResNet

def get_logger(name: str = __name__) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(name)-12s] [%(levelname)s]: %(message)s',
    )
    return logging.getLogger(name)

logger = get_logger(__name__)

def get_base_dir() -> Path:
    return Path(__file__).resolve().parent.parent

BASE_DIR = get_base_dir()
CONFIG_DIR = BASE_DIR / "configs"
DATASET_DIR = BASE_DIR / "datasets"
CKPT_DIR = BASE_DIR / "checkpoints"

def load_config(fname: str) -> Dict[str, Any]:
    cfg_path = CONFIG_DIR / fname
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    logger.info('Config loaded successfully')
    return cfg

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.info(f"Seed set to {seed}")

def load_datasets(cfg: Dict[str, Any]) -> tuple[DataLoader, DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(tuple(cfg['dataset']['mean']), tuple(cfg['dataset']['std']))
    ])

    if cfg['dataset']['name'].lower() == 'cifar':
        datasets_path = DATASET_DIR / cfg['dataset']['root']
        
        train_set = datasets.CIFAR10(
            root=datasets_path,
            train=True,
            transform=transform,
            download=True,
        )

        train_set, val_set = random_split(
            dataset=train_set,
            lengths=(45000, 5000),
        )

        test_set = datasets.CIFAR10(
            root=datasets_path,
            train=False,
            transform=transform,
            download=True,
        )

        train_loader = DataLoader(
            dataset=train_set,
            batch_size=cfg['train']['batch_size'],
            shuffle=True,
        )

        val_loader = DataLoader(
            dataset=val_set,
            batch_size=cfg['train']['batch_size'],
            shuffle=False,
        )

        test_loader = DataLoader(
            dataset=test_set,
            batch_size=cfg['train']['batch_size'],
            shuffle=False,
        )
    else:
        raise ValueError(f"Unsupported dataset: {cfg['dataset']['name']}")

    logger.info(f"Datasets loaded: train {len(train_loader.dataset)}, val {len(val_loader.dataset)}, test {len(test_loader.dataset)}")
    return train_loader, val_loader, test_loader

def load_model(cfg: Dict[str, Any]) -> ResNet:
    device = cfg['runtime']['device']

    model = ResNet(
        in_channels=cfg['model']['in_channels'],
        channels_per_layer=cfg['model']['channels_per_layer'],
        n=cfg['model']['n'],
        num_classes=cfg['model']['num_classes'],
    ).to(device)
    logger.info(f"Model loaded: {model.__class__.__name__} on {device}")
    return model

def load_optimizer(cfg: Dict[str, Any], model: torch.nn.Module) -> torch.optim.Optimizer:
    name = cfg['optim']['name'].lower()
    if name == 'adam':
        optim = torch.optim.Adam(
            model.parameters(),
            lr=cfg['optim']['lr'],
            weight_decay=cfg['optim'].get('weight_decay', 0.0),
        )
    else:
        raise ValueError(f"Unsupported optimizer: {name}")
    logger.info(f"Optimizer loaded: {cfg['optim']['name']} with lr={cfg['optim']['lr']}")
    return optim

def set_loss_fn(cfg: Dict[str, Any]) -> torch.nn.Module:
    name = cfg['train']['loss_fn'].lower()
    if name == 'cross_entropy':
        loss_fn = CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss function: {name}")
    logger.info(f"Loss function loaded: {cfg['train']['loss_fn']}")
    return loss_fn

def plot_loss(loss: list[float], title: str = None) -> None:
    _, ax = plt.subplots(1, 1)
    ax.plot(range(1, len(loss)+1), loss)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    if title:
        ax.set_title(title)
    plt.show()

def save_checkpoint(
    cfg: Dict[str, Any],
    epoch: int,
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    train_loss: list[float],
    val_loss: list[float] | None,
) -> None:
    checkpoint = {
        'seed': cfg['runtime']['seed'],
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_path = (
        CKPT_DIR /
        f"resnet{6*cfg['model']['n']+2}_"
        f"{cfg['dataset']['name']}_"
        f"{cfg['optim']['name']}_"
        f"epoch{epoch+1}_"
        f"{timestamp}.pth"
    )
    torch.save(checkpoint, ckpt_path)
    logger.info(f'Checkpoint saved: {ckpt_path}')

def load_checkpoint(
    fname: str,
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
) -> tuple[int, list[float], list[float] | None]:
    ckpt_path = CKPT_DIR / fname
    checkpoint = torch.load(ckpt_path, weights_only=False)
    seed = checkpoint['seed']
    model.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optimizer_state_dict'])
    train_loss = checkpoint['train_loss']
    val_loss = checkpoint['val_loss']
    return seed, train_loss, val_loss