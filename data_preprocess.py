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
        max_prob_len: int
        max_eq_len: int
        number_mappings: List[List]
    """
    problems, equations = [], []
    max_prob_len, max_eq_len = 0, 0
    with open(path, mode='r', encoding='utf-8') as f:
        data_list = json.load(f)
    for data in data_list:
        single_problem, single_equation = list(data["text"].split()), data["target_template"]
        # Get max length
        if max_prob_len <= len(single_problem):
            max_prob_len = len(single_problem)
        if max_eq_len <= len(single_equation):
            max_eq_len = len(single_equation)
        # Append
        problems.append(single_problem)
        equations.append(single_equation)
    return problems, equations, max_prob_len, max_eq_len


def pad_sequence(sequence, max_len):
    """Pad the sequence to the maximum length.

    Args:
        sequence: List[str]
        max_len: int
    """
    assert len(sequence) <= max_len
    return sequence + ['<pad>'] * (max_len - len(sequence))


def build_data(vocab, tokens, max_len):
    return torch.tensor([vocab[pad_sequence(line + ['<eos>'], max_len + 1)]
                         for line in tokens])  # max_len + 1 is because of <eos> token


set_seed()

train_path = './data/train.json'
valid_path = './data/valid.json'
test_path = './data/test.json'

# Add suffix to distinguish training set or test set.
src_tokens_train, tgt_tokens_train, max_prob_len_train, max_eq_len_train = read_data(train_path)
src_tokens_valid, tgt_tokens_valid, max_prob_len_valid, max_eq_len_valid = read_data(valid_path)
src_tokens_test, tgt_tokens_test, max_prob_len_test, max_eq_len_test = read_data(test_path)
max_prob_len = max(max_prob_len_train, max_prob_len_valid, max_prob_len_test)
max_eq_len = max(max_eq_len_train, max_eq_len_valid, max_eq_len_test)

src_vocab = Vocab(src_tokens_train + src_tokens_valid + src_tokens_test)
tgt_vocab = Vocab(tgt_tokens_train + tgt_tokens_valid + tgt_tokens_test)

src_data_train = build_data(src_vocab, src_tokens_train, max_len=max_prob_len)
tgt_data_train = build_data(tgt_vocab, tgt_tokens_train, max_len=max_eq_len)
src_data_valid = build_data(src_vocab, src_tokens_valid, max_len=max_prob_len)
tgt_data_valid = build_data(tgt_vocab, tgt_tokens_valid, max_len=max_eq_len)
src_data_test = build_data(src_vocab, src_tokens_test, max_len=max_prob_len)
tgt_data_test = build_data(tgt_vocab, tgt_tokens_test, max_len=max_eq_len)

train_data = TensorDataset(src_data_train, tgt_data_train)
valid_data = TensorDataset(src_data_valid, tgt_data_valid)
test_data = TensorDataset(src_data_test, tgt_data_test)

# Special tokens
PAD_IDX = src_vocab['<pad>']
BOS_IDX = src_vocab['<bos>']
EOS_IDX = src_vocab['<eos>']
