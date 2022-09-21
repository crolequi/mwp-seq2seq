import time
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_preprocess import *
from arch.transformer import Transformer, PositionalEncoding
from utils import equation_accuracy, s2hms


class Model(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 tgt_vocab_size,
                 d_model=512,
                 nhead=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 dim_feedforward=2048,
                 dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, dropout)

        self.transformer = Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)

        self.out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self,
                src,
                tgt,
                src_mask=None,
                tgt_mask=None,
                memory_mask=None,
                src_key_padding_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        """
        Args:
            src: (N, S)
            tgt: (N, T)
            tgt_mask: (T, T)
            src_key_padding_mask: (N, S)
            tgt_key_padding_mask: (N, T)
            memory_key_padding_mask: (N, S)
        """
        src = self.pe(self.src_embedding(src).transpose(0, 1) * math.sqrt(self.d_model))  # (S, N, E)
        tgt = self.pe(self.tgt_embedding(tgt).transpose(0, 1) * math.sqrt(self.d_model))  # (T, N, E)
        transformer_output = self.transformer(src=src,
                                              tgt=tgt,
                                              src_mask=src_mask,
                                              tgt_mask=tgt_mask,
                                              memory_mask=memory_mask,
                                              src_key_padding_mask=src_key_padding_mask,
                                              tgt_key_padding_mask=tgt_key_padding_mask,
                                              memory_key_padding_mask=memory_key_padding_mask)  # (T, N, E)
        logits = self.out(transformer_output)  # (T, N, tgt_vocab_size)
        return logits

    def encoder(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Args:
            src: (N, S)
        """
        src = self.pe(self.src_embedding(src).transpose(0, 1) * math.sqrt(self.d_model))
        memory = self.transformer.encoder(src, src_mask, src_key_padding_mask)
        return memory

    def decoder(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Args:
            tgt: (N, T)
        """
        tgt = self.pe(self.tgt_embedding(tgt).transpose(0, 1) * math.sqrt(self.d_model))
        decoder_output = self.transformer.decoder(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)
        logits = self.out(decoder_output)
        return logits


def train(train_loader, model, criterion, optimizer, scheduler, device):
    model.train()
    for batch_idx, (encoder_input, decoder_target) in enumerate(train_loader):
        encoder_input, decoder_target = encoder_input.to(device), decoder_target.to(device)
        bos_column = torch.tensor([BOS_IDX] * decoder_target.shape[0]).reshape(-1, 1).to(device)
        decoder_input = torch.cat((bos_column, decoder_target[:, :-1]), dim=1)

        tgt_mask = model.transformer.generate_square_subsequent_mask(max_tgt_len)
        src_key_padding_mask = encoder_input == PAD_IDX
        tgt_key_padding_mask = decoder_input == PAD_IDX

        pred = model(encoder_input,
                     decoder_input,
                     tgt_mask=tgt_mask.to(device),
                     src_key_padding_mask=src_key_padding_mask.to(device),
                     tgt_key_padding_mask=tgt_key_padding_mask.to(device),
                     memory_key_padding_mask=src_key_padding_mask.to(device))

        loss = criterion(pred.permute(1, 2, 0), decoder_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scheduler.step()

        if (batch_idx + 1) % 2 == 0:
            current, train_size = (batch_idx + 1) * encoder_input.size(0), len(train_loader.dataset)
            print(f"[{current:>5d}/{train_size:>5d}] train loss: {loss:.4f}")


@torch.no_grad()
def inference(test_loader, model, device):
    tgt_pred_equations = []
    model.eval()
    for src_seq, tgt_seq in test_loader:
        encoder_input = src_seq.to(device)
        src_key_padding_mask = encoder_input == PAD_IDX
        memory = model.encoder(encoder_input, src_key_padding_mask=src_key_padding_mask)
        pred_seq = [BOS_IDX]
        for _ in range(max_tgt_len):
            decoder_input = torch.tensor(pred_seq).reshape(1, -1).to(device)
            tgt_mask = model.transformer.generate_square_subsequent_mask(len(pred_seq))
            pred = model.decoder(decoder_input,
                                 memory,
                                 tgt_mask=tgt_mask.to(device),
                                 memory_key_padding_mask=src_key_padding_mask.to(device))  # (len(pred_seq), 1, tgt_vocab_size)
            next_token_idx = pred[-1].squeeze().argmax().item()
            if next_token_idx == EOS_IDX:
                break
            pred_seq.append(next_token_idx)
        pred_seq = tgt_vocab[pred_seq[1:]]  # Convert indices to string
        tgt_seq = tgt_seq.squeeze().tolist()
        tgt_seq = tgt_vocab[tgt_seq[:tgt_seq.index(EOS_IDX)]] if EOS_IDX in tgt_seq else tgt_vocab[tgt_seq]
        tgt_pred_equations.append((''.join(tgt_seq), ''.join(pred_seq)))
    return tgt_pred_equations


class LRScheduler:
    def __init__(self, optimizer, d_model=512, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.num_step = 1

        self.step()

    def step(self):
        new_lr = self.d_model**(-0.5) * min(self.num_step**(-0.5), self.num_step * self.warmup_steps**(-1.5))
        for group in self.optimizer.param_groups:
            group['lr'] = new_lr
        self.num_step += 1


# Parameter settings
set_seed()
BATCH_SIZE = 128
LEARNING_RATE = 0.0
NUM_EPOCHS = 100

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1)

# Model building
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Model(len(src_vocab), len(tgt_vocab)).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)
scheduler = LRScheduler(optimizer)

# Run
tic = time.time()
for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch + 1}\n" + "-" * 32)
    train(train_loader, model, criterion, optimizer, scheduler, device)
    print()

torch.save(model.state_dict(), './params/tfm.pt')
tgt_pred_equations = inference(test_loader, model, device)
equ_acc = equation_accuracy(tgt_pred_equations)
toc = time.time()

# Output
h, m, s = s2hms(toc - tic)
print("-" * 32 + f"\nEquation Accuracy: {equ_acc:.3f}\n" + "-" * 32)
print(f"{h}h {m}m {s}s")
