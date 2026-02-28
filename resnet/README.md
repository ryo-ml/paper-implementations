# ResNet (He et al., 2016)

This directory contains a PyTorch implementation of ResNet, following the original paper *“Deep Residual Learning for Image Recognition”*.

The implementation is designed for clarity, experimentation, and reproducibility, with explicit separation between training, evaluation, and model definitions.

---

## Features

- Modular training / validation / testing loops
- Separate entry points for training and evaluation
- Checkpoint save & load support
- CIFAR-10 and ImageNet-1k ResNet architectures

---

## Directory Structure

```text
resnet/
├── main.py              # Entry point for training + validation + testing
├── eval.py              # Entry point for evaluation using saved checkpoints
├── train.py             # train_one_epoch, val_one_epoch, test_one_epoch
├── model_cifar.py       # ResNet model definition for CIFAR-10
├── model_imagenet.py    # ResNet model definition for ImageNet-1k
├── utils.py             # Logging, config, checkpoint utilities
├── configs/             # Experiment configuration files
│   └── resnet20_cifar.yaml
├── datasets/            # Datasets (ignored by git)
├── checkpoints/         # Saved model checkpoints (ignored by git)
└── README.md
```

---

## Usage

### Training

```bash
python -m resnet.src.main
```

This runs:

- training
- periodic validation
- final testing
- checkpoint saving

---

### Evaluation

```bash
python -m resnet.src.eval
```

Evaluation is performed using a saved checkpoint without re-training.

---

## Notes

- `datasets/` and `checkpoints/` are intentionally excluded from version control
- All experiment settings should be controlled via YAML config files
- This code is intended as a paper implementation and not optimized for performance
- ImageNet ResNet models are provided only for architectural reference
- Training and evaluation have been validated on CIFAR-10

---

## References

Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.  
Deep Residual Learning for Image Recognition.  
*CVPR*, 2016.