from tqdm import tqdm

from .utils import *
from .train import *

logger = get_logger(__name__)

# load config
cfg = load_config('resnet20_cifar.yaml')

# set seed
set_seed(cfg['runtime']['seed'])

# load datasets
train_loader, val_loader, test_loader = load_datasets(cfg)

# load model 
model = load_model(cfg)

# load optimizer
optim = load_optimizer(cfg, model)

# set loss function
loss_fn = set_loss_fn(cfg)

# train
train_loss = []
val_loss = []
for epoch_num in tqdm(range(cfg['train']['epochs']), desc='Train (whole)'):
    train_batch_mean_loss = train_one_epoch(
        cfg,
        train_loader,
        model,
        optim,
        loss_fn,
        epoch_num,
    )
    train_loss.append(train_batch_mean_loss)

    # validaion
    if (epoch_num + 1) % cfg['train']['val_freq'] == 0:
        val_acc, val_batch_mean_loss = val_one_epoch(
            cfg,
            val_loader,
            model,
            loss_fn,
            epoch_num,
        )
        val_loss.append(val_batch_mean_loss)

# test
test_acc, test_loss = test_one_epoch(
    cfg,
    test_loader,
    model,
    loss_fn,
    epoch_num,
)

# plot
plot_loss(train_loss, 'Train')

# save checkpoint
save_checkpoint(
    cfg,
    epoch_num,
    model,
    optim,
    train_loss,
    val_loss,    
)