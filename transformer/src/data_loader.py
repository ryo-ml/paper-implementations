from typing import Any
import datasets
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from .tokenizer import AbstractTokenizer
from .logger import get_logger

logger = get_logger(__name__)

def _preprocess_dataset(dataset: dict[str, Any]) -> dict[str, Any]:
    '''
    Preprocesses dataset. The operation includes concatenation and extraction of answer texts
    '''
    def _preprocess_function(examples: dict[str, list]) -> dict[str, list]:
        '''
        Preprocessing function for `datasets.Dataset.map`
        '''
        sources = []
        targets = []

        for personalities, utterances in zip(examples['personality'], examples['utterances']):
            persona_prefix = " ".join(personalities)

            for utterance in utterances:
                history = utterance['history']
                reply = utterance['candidates'][-1]

                source_text = f"{persona_prefix} {' '.join(history)}"

                sources.append(source_text)
                targets.append(reply)
        
        return {
            'source': sources,
            'target': targets
        }

    preprocessed_dataset = dataset.map(
        _preprocess_function,
        batched=True,
        remove_columns=dataset['train'].column_names,
        desc='Preprocessing',
    )

    return preprocessed_dataset

def _tokenize_dataset(dataset: dict[str, Any], tokenizer: AbstractTokenizer | None = None, max_len: int = 512) -> dict[str, Any]:
    '''
    Tokenizes dataset. 
    '''
    if tokenizer is None:
        raise ValueError('Tokenizer is required')

    def _tokenize_function(examples: dict[str, list]) -> dict[str, list]:
        '''
        Tokenizing function for `datasets.Dataset.map`
        '''
        input_ids = [tokenizer.encode(text, add_special_tokens=True)[:max_len] for text in examples['source']]
        labels = [tokenizer.encode(text, add_special_tokens=True)[:max_len] for text in examples['target']]

        return {
            'input_ids': input_ids,
            'labels': labels
        }

    tokenized_dataset = dataset.map(
        _tokenize_function,
        batched=True,
        remove_columns=dataset['train'].column_names,
        desc='Tokenizing',
    )

    return tokenized_dataset

def load_dataset(cfg: dict[str, Any], tokenizer: AbstractTokenizer | None = None, max_len: int = 512) -> dict[str, Any]:
    '''
    Loads the dataset, applying preprocessing and tokenization.
    '''
    dataset = datasets.load_dataset(cfg['dataset']['name'])
    preprocessed_dataset = _preprocess_dataset(dataset)
    tokenized_dataset = _tokenize_dataset(preprocessed_dataset, tokenizer, max_len)
    logger.info("Dataset loading and pipeline finished successfully")

    return tokenized_dataset

def create_dataloaders(dataset: dict[str, Any], batch_size: int, padding_token_id: int = 0) -> tuple[dict[str, list], dict[str, list]]:
    '''
    Creates dataloader.
    '''
    def _collate_function(batch: list[dict[str, Any]]):
        src_list = [torch.tensor(item['input_ids']) for item in batch]
        tgt_list = [torch.tensor(item['labels']) for item in batch]

        src_padded = pad_sequence(
            src_list,
            batch_first=True,
            padding_value=padding_token_id
        )

        tgt_padded = pad_sequence(
            tgt_list,
            batch_first=True,
            padding_value=padding_token_id
        )

        decoder_input_ids = tgt_padded[:, :-1].clone()
        labels = tgt_padded[:, 1:].clone()

        return {
            'input_ids': src_padded,
            'decoder_input_ids': decoder_input_ids,
            'labels': labels
        }

    train_loader = DataLoader(
        dataset['train'],
        batch_size,
        shuffle=True,
        collate_fn=_collate_function,
    )

    val_loader = DataLoader(
        dataset['validation'],
        batch_size,
        shuffle=False,
        collate_fn=_collate_function,
    )
    logger.info(f"Dataloaders created successfully")

    return train_loader, val_loader
