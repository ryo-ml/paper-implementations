from pathlib import Path
from tqdm import tqdm
import torch

from .config import *
from .tokenizer import *
from .data_loader import *
from .builder import *
from .train import *
from .logger import get_logger

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / 'configs'
CKPT_DIR = BASE_DIR / 'checkpoints'

logger = get_logger(__name__)

cfg = load_config(CONFIG_DIR / 'config.yaml')
device = cfg['device']
vocab_size = cfg['model']['vocab_size']
batch_size = cfg['train']['batch_size']
epochs = cfg['train']['epochs']

ckpt_path: Path = CKPT_DIR / cfg['train']['checkpoint']['fname']
if ckpt_path.exists():
    raise FileExistsError('Checkpoint already exists. Change the path to prevent overwrite')

set_seed(cfg['seed'])
tokenizer = build_tokenizer(cfg)
dataset = load_dataset(cfg, tokenizer, max_len=cfg['model']['max_len']) # TODO remove max_len
train_loader, _ = create_dataloaders(dataset, batch_size, tokenizer.vocab['<PAD>'])

model = build_model(
    cfg, 
    tokenizer.vocab['<PAD>'],
    tokenizer.vocab['<BOS>'],
    tokenizer.vocab['<EOS>'],
)
optim = build_optimizer(cfg, model)
scheduler = build_scheduler(cfg, optim)
loss_fn = build_loss_fn(cfg, tokenizer.vocab['<PAD>'])

epoch_loss = []
for epoch in tqdm(range(epochs)):
    batch_loss = train_one_epoch(
        train_loader,
        model,
        optim,
        scheduler, 
        loss_fn,
        epoch,
        device,
    )
    epoch_loss.append(batch_loss)

save_checkpoint(
    cfg=cfg,
    model=model,
    optim=optim,
    scheduler=scheduler,
    epoch=epoch,
    loss=epoch_loss,
)

input_text = 'My favorite thing is to hunt cheetahs'
input_ids = tokenizer.encode(input_text, add_special_tokens=True)
input_ids = torch.tensor([input_text], device=device)
token_ids = infer(model, input_ids)
output_text = tokenizer.decode(token_ids[0], skip_special_tokens=True)
print()
print(f'input: {input_text}')
print(f'output: {output_text}')
