import time
import torch
import torch.nn as nn

from data_preprocess import *
from torch.utils.data import DataLoader
from utils import equation_accuracy, s2hms


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size=256, num_layers=2, dropout=0.5):
        super().__init__()
        self.emebdding = nn.Embedding(vocab_size, embed_size, padding_idx=PAD_IDX)
        self.rnn = nn.GRU(embed_size, hidden_size, num_layers=num_layers, dropout=dropout)

    def forward(self, encoder_input):
        """
        Args:
            encoder_input: shape (batch_size, seq_len)
        """
        encoder_input = self.emebdding(encoder_input).transpose(0, 1)
        _, h_n = self.rnn(encoder_input)
        c_n = torch.zeros_like(h_n)  # The initial hidden state of the LSTM requires c_n.
        return h_n, c_n


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size=256, num_layers=2, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=PAD_IDX)
        self.rnn = nn.LSTM(embed_size + hidden_size, hidden_size, num_layers=num_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, encoder_output, decoder_input):
        h_n, c_n = encoder_output
        decoder_input = self.embedding(decoder_input).transpose(0, 1)
        context = h_n[-1]  # (batch_size, hidden_size)
        context = context.repeat(decoder_input.size(0), 1, 1)  # (seq_len, batch_size, hidden_size)
        output, (h_n, c_n) = self.rnn(torch.cat((decoder_input, context), -1), (h_n, c_n))
        logits = self.out(output)  # (seq_len, batch_size, vocab_size)
        return logits, (h_n, c_n)


class Model(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, encoder_input, decoder_input):
        return self.decoder(self.encoder(encoder_input), decoder_input)


def train(train_loader, model, criterion, optimizer, device):
    model.train()
    for batch_idx, (encoder_input, decoder_target) in enumerate(train_loader):
        encoder_input, decoder_target = encoder_input.to(device), decoder_target.to(device)
        bos_column = torch.tensor([BOS_IDX] * decoder_target.shape[0]).reshape(-1, 1).to(device)
        decoder_input = torch.cat((bos_column, decoder_target[:, :-1]), dim=1)
        pred, (_, _) = model(encoder_input, decoder_input)
        loss = criterion(pred.permute(1, 2, 0), decoder_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 2 == 0:
            current, train_size = (batch_idx + 1) * encoder_input.size(0), len(train_loader.dataset)
            print(f"[{current:>5d}/{train_size:>5d}] train loss: {loss:.4f}")


@torch.no_grad()
def inference(test_loader, model, device):
    tgt_pred_equations = []
    model.eval()
    for src_seq, tgt_seq in test_loader:
        encoder_input = src_seq.to(device)
        h_n, c_n = model.encoder(encoder_input)
        pred_seq = [BOS_IDX]
        for _ in range(max_tgt_len):
            decoder_input = torch.tensor(pred_seq[-1]).reshape(1, 1).to(device)  # (batch_size, seq_len)=(1, 1)
            pred, (h_n, c_n) = model.decoder((h_n, c_n),
                                             decoder_input)  # pred shape: (seq_len, batch_size, tgt_vocab_size)=(1, 1, tgt_vocab_size)
            next_token_idx = pred.squeeze().argmax().item()
            if next_token_idx == EOS_IDX:
                break
            pred_seq.append(next_token_idx)
        pred_seq = tgt_vocab[pred_seq[1:]]  # Convert indices to string
        tgt_seq = tgt_seq.squeeze().tolist()
        tgt_seq = tgt_vocab[tgt_seq[:tgt_seq.index(EOS_IDX)]] if EOS_IDX in tgt_seq else tgt_vocab[tgt_seq]
        tgt_pred_equations.append((''.join(tgt_seq), ''.join(pred_seq)))
    return tgt_pred_equations


class LRScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.num_epoch = 0
        self.step()

    def step(self):
        if self.num_epoch <= 20:
            new_lr = 0.01
        elif 20 < self.num_epoch <= 60:
            new_lr = 0.001
        else:
            new_lr = 0.0001
        for group in self.optimizer.param_groups:
            group["lr"] = new_lr
        self.num_epoch += 1


# Parameter settings
set_seed()
BATCH_SIZE = 128
LEARNING_RATE = 0.01
NUM_EPOCHS = 100

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1)

# Model building
device = 'cuda' if torch.cuda.is_available() else 'cpu'
encoder = Encoder(vocab_size=len(src_vocab), embed_size=256)
decoder = Decoder(vocab_size=len(tgt_vocab), embed_size=256)
model = Model(encoder, decoder).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = LRScheduler(optimizer)

# Run
tic = time.time()
for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch + 1}\n" + "-" * 32)
    train(train_loader, model, criterion, optimizer, device)
    scheduler.step()
    print()

torch.save(model.state_dict(), './params/vanilla.pt')
tgt_pred_equations = inference(test_loader, model, device)
equ_acc = equation_accuracy(tgt_pred_equations, verbose=True)
toc = time.time()

# Output
h, m, s = s2hms(toc - tic)
print("-" * 32 + f"\nEquation Accuracy: {equ_acc:.3f}\n" + "-" * 32)
print(f"{h}h {m}m {s}s")
