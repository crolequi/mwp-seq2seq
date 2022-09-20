import json
import torch

from torch.utils.data import TensorDataset
from utils import Vocab, set_seed


def read_data(path):
    """Read problems and equations from json file and tokenize.

    Args:
        path: A string representing the storage path of the json file.

    Returns:
        problems: List[List]
        equations: List[List]
    """
    problems, equations = [], []
    with open(path, mode='r', encoding='utf-8') as f:
        data_list = json.load(f)
    for data in data_list:
        problems.append(list(data["text"].split()))
        equations.append(list(data["template_equ"].split()))
    return problems, equations


def pad_sequence(sequence, max_len):
    """Pad the sequence to the maximum length.

    Args:
        sequence: List[str]
        max_len: int
    """
    assert len(sequence) <= max_len
    return sequence + ['<pad>'] * (max_len - len(sequence))


def build_data(vocab, tokens, max_len):
    return torch.tensor([vocab[pad_sequence(line + ['<eos>'], max_len)] for line in tokens])  # max_len + 1 is because of <eos> token


set_seed()
train_path, test_path = './data/train.json', './data/test.json'

# Add suffix to distinguish training set or test set.
src_tokens_train, tgt_tokens_train = read_data(train_path)
src_tokens_test, tgt_tokens_test = read_data(test_path)
src_tokens = src_tokens_train + src_tokens_test
tgt_tokens = tgt_tokens_train + tgt_tokens_test

max_src_len, max_tgt_len = 80, 30

src_vocab, tgt_vocab = Vocab(src_tokens), Vocab(tgt_tokens)

src_data_train = build_data(src_vocab, src_tokens_train, max_len=max_src_len)
tgt_data_train = build_data(tgt_vocab, tgt_tokens_train, max_len=max_tgt_len)
src_data_test = build_data(src_vocab, src_tokens_test, max_len=max_src_len)
tgt_data_test = build_data(tgt_vocab, tgt_tokens_test, max_len=max_tgt_len)

train_data = TensorDataset(src_data_train, tgt_data_train)
test_data = TensorDataset(src_data_test, tgt_data_test)

# Special tokens
PAD_IDX = src_vocab['<pad>']
BOS_IDX = src_vocab['<bos>']
EOS_IDX = src_vocab['<eos>']
