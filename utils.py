import random
import string

import torch
import torch.nn as nn
import numpy as np

from collections import Counter
from tkinter import _flatten


class Vocab:
    def __init__(self, tokens, min_count=1, special_tokens=['<unk>', '<pad>', '<bos>', '<eos>']):
        """Building vocabulary from tokens.

        Args:
            tokens: A 2D list, where each list in tokens corresponds to a sentence.
            min_count: Occurrences less than min_count are discarded and treated as <unk>.
        """
        assert min_count > 0 and '<unk>' in special_tokens
        self.tokens = tokens
        self.min_count = min_count

        self.token2idx = {token: idx for idx, token in enumerate(special_tokens)}
        self.token2idx.update({
            token: idx + len(special_tokens)
            for idx, (token, count) in enumerate(sorted(Counter(_flatten(self.tokens)).items(), key=lambda x: x[1], reverse=True))
            if count >= self.min_count
        })
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}

    def __getitem__(self, tokens_or_indices):
        """Get elements in the vocabulary, support forward and reverse search.

        Args:
            tokens_or_indices: A single token or a single index or a list of tokens or a list of indices.
        """
        if isinstance(tokens_or_indices, (str, int)):
            return self.token2idx.get(tokens_or_indices, self.token2idx['<unk>']) if isinstance(
                tokens_or_indices, str) else self.idx2token.get(tokens_or_indices, '<unk>')
        elif isinstance(tokens_or_indices, list):
            return [self.__getitem__(item) for item in tokens_or_indices]
        else:
            raise TypeError

    def __len__(self):
        return len(self.idx2token)


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def equation_accuracy(tgt_pred_equations, verbose=False):
    """
    Args:
        tgt_pred_equations: List[Tuple[str, str]]
    """
    correct = 0
    for tgt, pred in tgt_pred_equations:
        if tgt == pred:
            correct += 1
    return correct / len(tgt_pred_equations)


def s2hms(s):
    """Convert seconds to hours, minutes and seconds format

    Args:
        s: seconds, int or str
    """
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return h, m, s
