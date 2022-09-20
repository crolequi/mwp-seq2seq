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


class RuleFilter(nn.Module):
    def __init__(self, tgt_vocab):
        super().__init__()
        self.tgt_vocab = tgt_vocab

    def forward(self, logits):
        """Filter out illegal tokens in logits based on rules.

        Args:
            logits: shape (L, N, tgt_vocab_size), where L is sequence length, N is batch size.
        """
        L = logits.size(0)
        assert L >= 2
        for i in range(1, L):
            logits[i] = self.batch_filter(logits[i - 1], logits[i])
        return logits

    def batch_filter(self, prev, cur):
        """
        Args:
            prev: (batch_size, tgt_vocab_size)
            cur: (batch_size, tgt_vocab_size)
        """
        for i in range(len(cur)):
            cur[i] = self.single_filter(prev[i], cur[i])
        return cur

    def single_filter(self, prev, cur):
        """Filter out illegal tokens.

        Args:
            prev: shape (tgt_vocab_size, ) or (), the logits output by the neural network or a single index.
            cur: shape (tgt_vocab_size, ), the logits output by the neural network.
        """
        rho = torch.ones(len(self.tgt_vocab)).to(prev.device)
        if prev.dim() == 1:
            prev_token = self.tgt_vocab[prev.argmax().item()]
        elif prev.dim() == 0:
            prev_token = self.tgt_vocab[prev.item()]
        else:
            raise RuntimeError

        # Rule 1
        if prev_token in "+-*/":
            rho[self.tgt_vocab[list("+-*/)=")]] = 0  # Set the position corresponding to the illegal token to 0
        # Rule 2
        elif "temp" in prev_token:
            rho[self.tgt_vocab[list("(=")]] = 0
        # Rule 3
        elif prev_token == "=":
            rho[self.tgt_vocab[list("+-*/=)") + [f"temp_{alpha}" for alpha in string.ascii_lowercase]]] = 0
        # Rule 4
        elif prev_token == "(":
            rho[self.tgt_vocab[list("()+-*/=")]] = 0
        # Rule 5
        elif prev_token == ")":
            rho[self.tgt_vocab[list("()") + [f"temp_{alpha}" for alpha in string.ascii_lowercase]]] = 0
        # Additional Rule
        elif prev_token == "^":
            rho[self.tgt_vocab[list("+-*/=)")]] = 0
        else:
            pass

        # Element-wise product
        cur = cur * rho
        return cur


def equation_accuracy(tgt_pred_equations, verbose=False):
    """
    Args:
        tgt_pred_equations: List[Tuple[str, str]]
    """
    correct = 0
    for tgt, pred in tgt_pred_equations:
        if verbose:
            print(tgt, "|", pred)
        if tgt == pred:
            correct += 1
    if verbose:
        print()
    return correct / len(tgt_pred_equations)


def s2hms(s):
    """Convert seconds to hours, minutes and seconds format

    Args:
        s: seconds, int or str
    """
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return h, m, s
