# Transformer (Vaswani et al., 2017)

This directory contains a PyTorch implementation of Transformer, following the original paper *"Attention Is All You Need"*.

---

## Features

- The model is trained and evaluated on the Persona-Chat dataset.
- Full Encoder-Decoder architecture
- Scaled Dot-Product Multi-Head Attention implementation
- Positional Encoding (Sinusoidal)
- Noam Scheduler support

---

## Directory Structure

```text
transformer/
├── src/
│   ├── main.py          # Entry point for training
│   ├── train.py         # Training and validation logic
│   ├── eval.py          # Entry point for evaluation
│   ├── model.py         # Transformer architecture (Encoder/Decoder)
│   ├── tokenizer.py     # Tokenization wrappers (Word/BPE)
│   ├── data_loader.py   # Dataset handling and batching
│   ├── mask.py          # Padding and look-ahead mask generation
│   ├── builder.py       # Helper for building tokenizer, model,  optimizer, loss, scheduler, and checkpoint load/save.
│   ├── config.py        # Helper for configuration
│   └── logger.py        # Training progress logging
├── configs/             # Experiment configuration files (e.g., config.yaml)
├── tokenizers/          # Saved tokenizer states (e.g., bpe.json) - Ignored by git
├── checkpoints/         # Model weights (.pth) - Ignored by git
└── README.md
```

---

## Usage

Run all commands from the `transformer/` root directory:

### Training

```bash
python -m src.main
```

This runs:

- training
- checkpoint saving

---

### Evaluation

```bash
python -m src.eval
```

Evaluation is performed using a saved checkpoint without re-training.

---

## Notes

- `datasets/` and `checkpoints/` are intentionally excluded from version control
- All experiment settings should be controlled via YAML config files
- This code is intended as a paper implementation and not optimized for performance

---

## References

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin.

Attention Is All You Need.

*NeurIPS*, 2017.
