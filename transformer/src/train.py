from typing import Any
from pathlib import Path
from tqdm import tqdm
import torch

from .logger import get_logger

logger = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent

def train_one_epoch(
    train_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,    
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    loss_fn: torch.nn.Module,
    epoch: int,
    device: torch.device,
) -> list[float]:
    model.train()

    batch_loss = []
    pbar = tqdm(train_loader, desc=f'Train Epoch {epoch}')

    for batch_idx, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        decoder_input_ids = batch['decoder_input_ids'].to(device)
        labels = batch['labels'].to(device)

        optim.zero_grad()
        logits = model(
            src=input_ids,
            tgt=decoder_input_ids,
        )

        loss = loss_fn(
            logits.view(-1, logits.size(2)),
            labels.view(-1)
        )
        loss.backward()
        optim.step()
        scheduler.step()

        batch_loss.append(loss.item())
        
        pbar.set_postfix({
            'loss': f'{loss.item():.6f}',
            'lr': f"{optim.param_groups[0]['lr']: .2e}",
        })

    return batch_loss

def val_one_epoch(
    val_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    epoch: int,
    device: torch.device,
) -> list[float]:
    model.eval()

    batch_loss = []
    pbar = tqdm(val_loader, desc=f'Val Epoch {epoch}')

    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            decoder_input_ids = batch['decoder_input_ids'].to(device)
            labels = batch['labels'].to(device)

            logits = model(
                src=input_ids,
                tgt=decoder_input_ids,
            )

            loss = loss_fn(
                logits.view(-1, logits.size(2)),
                labels.view(-1)
            )

            batch_loss.append(loss.item())

            pbar.set_postfix({
                'loss': f'{loss.item():.6f}',
            })

    return batch_loss

def infer(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        token_ids = model.generate(input_ids)
    return token_ids

    # TODO change search algorithm (e.g., beam search)
