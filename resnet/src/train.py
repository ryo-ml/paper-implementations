from typing import Dict, Any
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

from .utils import get_logger

logger = get_logger(__name__)

def train_one_epoch(
    cfg: Dict[str, Any],
    train_loader: DataLoader,
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    current_epoch: int,
) -> float:
    device = cfg['runtime']['device']
    model.train()
    train_batch_loss = []

    for batch_num, (X, y) in enumerate(tqdm(train_loader, desc='Train', leave=False)):
        X, y = X.to(device), y.to(device)

        optim.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, y)
        train_batch_loss.append(loss.item())
        loss.backward()
        optim.step()
    train_batch_mean_loss =  np.mean(train_batch_loss)
    logger.info(f"Epoch {current_epoch+1}/{cfg['train']['epochs']} Loss: {train_batch_mean_loss:.4f}")
    return train_batch_mean_loss

def val_one_epoch(
    cfg: Dict[str, Any],
    val_loader: DataLoader,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module = None,
    current_epoch: int = None,
) -> tuple[float, float]:
    device = cfg['runtime']['device']
    model.eval()
    correct = 0
    val_batch_loss = []

    for batch_num, (X, y) in enumerate(tqdm(val_loader, desc='Val', leave=False)):
        X, y = X.to(device), y.to(device)

        with torch.no_grad():
            logits = model(X)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == y).sum().item()

            if loss_fn is not None:
                loss = loss_fn(logits, y)
                val_batch_loss.append(loss.item())

    val_acc = correct / len(val_loader.dataset) * 100
    val_batch_mean_loss = np.mean(val_batch_loss) if loss_fn is not None else None

    msg = []
    if current_epoch is not None:
        msg.append(f"Epoch {current_epoch+1}/{cfg['train']['epochs']}")
    msg.append(f"Val Acc: {val_acc:.2f}%")
    if loss_fn is not None:
        msg.append(f"Val Loss: {val_batch_mean_loss:.4f}")
    logger.info(" ".join(msg))
    return val_acc, val_batch_mean_loss

def test_one_epoch(
    cfg: Dict[str, Any],
    test_loader: DataLoader,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module = None,
    current_epoch: int = None,
) -> tuple[float, float]:
    device = cfg['runtime']['device']
    model.eval()
    correct = 0
    test_batch_loss = []

    for batch_num, (X, y) in enumerate(tqdm(test_loader, desc='Test', leave=False)):
        X, y = X.to(device), y.to(device)

        with torch.no_grad():
            logits = model(X)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == y).sum().item()

            if loss_fn is not None:
                loss = loss_fn(logits, y)
                test_batch_loss.append(loss.item())

    test_acc = correct / len(test_loader.dataset) * 100
    test_batch_mean_loss = np.mean(test_batch_loss) if loss_fn is not None else None

    msg = []
    if current_epoch is not None:
        msg.append(f"Epoch {current_epoch+1}/{cfg['train']['epochs']}")
    msg.append(f"Test Acc: {test_acc:.2f}%")
    if loss_fn is not None:
        msg.append(f"Test Loss: {test_batch_mean_loss:.4f}")
    logger.info(" ".join(msg))
    return test_acc, test_batch_mean_loss