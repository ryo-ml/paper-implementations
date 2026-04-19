from abc import ABC, abstractmethod
from collections import Counter
import json
from pathlib import Path
import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


class AbstractTokenizer(ABC):
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.vocab = {} # {'token': token_id}
        self.reverse_vocab = {} # {'token_id': token}

        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3,
        }
        self.vocab.update(self.special_tokens)
        # update reverse vocab
        self._update_reverse_vocab()

    def __len__(self):
        return len(self.vocab)

    def _update_reverse_vocab(self) -> None:
        self.reverse_vocab = {
            token_id: token for token, token_id in self.vocab.items()
        }

    @abstractmethod
    def train(self, corpus: list[str]) -> None:
        '''
        Construct vocabulary from corpus
        '''
        pass

    @abstractmethod
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        pass

    @abstractmethod
    def decode(self, token_ids: torch.Tensor | list[int], skip_special_tokens: bool = True) -> str:
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Path):
        pass

class WordTokenizer(AbstractTokenizer):
    def __init__(self, vocab_size: int = 10000):
        super().__init__(vocab_size)

    def train(self, corpus: list[str]) -> None:
        word_counts = Counter()
        for text in corpus:
            word_counts.update(text.split())

        # self.vocab
        num_common = self.vocab_size - len(self.special_tokens)
        for word, _ in word_counts.most_common(num_common):
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)

        # self.reverse_vocab
        self._update_reverse_vocab()

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        tokens = text.split()
        token_ids = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]

        if add_special_tokens:
            token_ids = [self.vocab['<BOS>']] + token_ids + [self.vocab['<EOS>']]

        return token_ids

    def decode(self, token_ids: torch.Tensor | list[int], skip_special_tokens: bool = True) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().numpy().tolist()
            
        tokens = []
        for token_id in token_ids:
            if token_id == self.vocab['<EOS>']:
                if not skip_special_tokens:
                    tokens.append(self.reverse_vocab[token_id])
                break

            if skip_special_tokens and token_id in self.special_tokens.values():
                continue

            tokens.append(self.reverse_vocab.get(token_id, '<UNK>'))

        return ' '.join(tokens)

    def save(self, path: Path) -> None:
        if path.exists():
            raise FileExistsError(f'File already exists: {path}')
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump({
                'vocab': self.vocab,
                'vocab_size': self.vocab_size,
            }, f)

    @classmethod
    def load(cls, path: Path) -> AbstractTokenizer:
        with open(path, 'r') as f:
            data = json.load(f)

        tokenizer = cls(vocab_size=data['vocab_size'])
        tokenizer.vocab = data['vocab']
        tokenizer._update_reverse_vocab()
        return tokenizer
        
class BytePairEncoding(AbstractTokenizer):
    def __init__(self, vocab_size: int = 10000):
        super().__init__(vocab_size)
        self.hf_tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
        self.hf_tokenizer.pre_tokenizer = Whitespace()

    def train(self, corpus: list[str]) -> None:
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        )
        self.hf_tokenizer.train_from_iterator(corpus, trainer)
        
        self.vocab = self.hf_tokenizer.get_vocab()
        self._update_reverse_vocab()

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = self.hf_tokenizer.encode(text).ids
        if add_special_tokens:
            ids = [self.vocab['<BOS>']] + ids + [self.vocab['<EOS>']]
        return ids

    def decode(self, token_ids: torch.Tensor | list[int], skip_special_tokens: bool = True) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().numpy().tolist()
        return self.hf_tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def save(self, path: Path) -> None:
        if path.exists():
            raise FileExistsError(f'File already exists: {path}')
        path.parent.mkdir(parents=True, exist_ok=True)

        self.hf_tokenizer.save(str(path))

    @classmethod
    def load(cls, path: Path) -> AbstractTokenizer:
        if not path.exists():
            raise FileNotFoundError(f'File not found: {path}')

        hf_tokenizer = Tokenizer.from_file(str(path))
        vocab_size = len(hf_tokenizer.get_vocab())

        tokenizer = cls(vocab_size=vocab_size)
        tokenizer.hf_tokenizer = hf_tokenizer

        tokenizer.vocab = tokenizer.hf_tokenizer.get_vocab()
        tokenizer._update_reverse_vocab()

        return tokenizer


if __name__ == '__main__':
    text = 'Slavery is freedom.'
    corpus = [text, "War is peace.", "Ignorance is strength."]

    tokenizer = BytePairEncoding(vocab_size=10000)
    print(type(tokenizer))
    tokenizer.train(corpus)
    token_ids = tokenizer.encode(text)
    tokens = tokenizer.decode(token_ids)
    print(token_ids)
    print(tokens)
