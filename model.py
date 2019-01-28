import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, nlayers, dropout):
        super().__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.nlayers = nlayers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, nlayers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, out_dim, emb_dim, hidden_dim, nlayers, dropout):
        super().__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.nlayers = nlayers
        self.dropout = dropout

        self.embedding = nn.Embedding(out_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, nlayers, dropout=dropout)
        self.out = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputx, hidden, cell):
        inputx = inputx.unsqueeze(0)
        embedded = self.dropout(self.embedding(inputx))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.out(output.squeeze(0))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hidden_dim == decoder.hidden_dim, 'Hidden dimension of encoder and decoder must be equal!'
        assert encoder.nlayers == decoder.nlayers, 'nlayers of the encoder and decoder must be equal!'

    def forward(self, src, trg, teacher_forcing_ratio =0.5):
        bsize = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.out_dim

        outputs = torch.zeros(max_len, bsize, trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(src)
        inputz = trg[0,:]
        for t in range(1, max_len):
            output, hidden, cell = self.decoder(inputz, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            inputz = (trg[t] if teacher_force else top1)
        return outputs

         
