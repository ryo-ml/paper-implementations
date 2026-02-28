from tqdm import tqdm

from .utils import *
from .train import test_one_epoch

logger = get_logger(__name__)

# load config
cfg = load_config('resnet20_cifar.yaml')

# load datasets
train_loader, val_loader, test_loader = load_datasets(cfg)

# load model
model = load_model(cfg)

# load optimizer
optim = load_optimizer(cfg, model)

# set loss function
loss_fn = set_loss_fn(cfg)

# load checkpoint
_, _, _ = load_checkpoint('resnet20_cifar_adam_epoch5_20260228_161024.pth', model, optim)

test_one_epoch(
    cfg,
    test_loader,
    model,
    loss_fn,
)