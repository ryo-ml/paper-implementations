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

set_seed(cfg['seed'])
tokenizer = build_tokenizer(cfg)
dataset = load_dataset(cfg, tokenizer, max_len=cfg['model']['max_len']) # TODO remove max_len
_, val_loader = create_dataloaders(dataset, batch_size, tokenizer.vocab['<PAD>'])

model = build_model(
    cfg, 
    tokenizer.vocab['<PAD>'],
    tokenizer.vocab['<BOS>'],
    tokenizer.vocab['<EOS>'],
)

model, _, _, epoch, loss = load_checkpoint(
    cfg=cfg,
    model=model,
    optim=None,
    scheduler=None,
)

input_text = 'What do you do in your free time?'
input_ids = tokenizer.encode(input_text, add_special_tokens=True)
input_ids = torch.tensor([input_ids], device=device)
token_ids = infer(model, input_ids)
output_text = tokenizer.decode(token_ids[0], skip_special_tokens=True)
print()
print(f'input: {input_text}')
print(f'output: {output_text}')
