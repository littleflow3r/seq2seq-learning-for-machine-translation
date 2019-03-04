import torch
import torch.nn as nn
import random

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout):
        super().__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.dropout = dropout

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.gru = nn.GRU(emb_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq):
        #seq = [seq len, batch size]
        embedded = self.dropout(self.embedding(seq)) # out shape [seq len, batch size, emb dim]
        outputs, hidden = self.gru(embedded)
        #outputs = [seq len, batch size, hid dim*n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        return hidden

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.gru = nn.GRU(emb_dim+hid_dim, hid_dim) #emb_dim is current input (target)+hid_dim is context vector from encoder
        self.out = nn.Linear(emb_dim+hid_dim*2, output_dim) #hid dim*2 = context vector encoder + prev decoder hidden states 
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input, hidden, context):
        #input = [batch size]
        #hidden = [n layers * n directions, batch size, hid dim]
        #context = [n layers * n directions, batch size, hid dim]
        input = input.unsqueeze(0) # [1, batch size]

        embedded = self.dropout(self.embedding(input)) # [1, batch size, emb dim]
        emb_concat = torch.cat((embedded, context), dim=2)
        
        output, hidden = self.gru(emb_concat, hidden) # decoder's first hidden state h0 is from encoder last hidden state
        output = torch.cat((embedded.squeeze(0), hidden.squeeze(0), context.squeeze(0)), dim=1)
        pred = self.out(output)
        return pred, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert encoder.hid_dim == decoder.hid_dim, "hidden dimensions of encoder and decoder must be equal!"

    def forward(self, src, trg=None, teacher_forcing_ratio=0.5):
        if trg is None:
            trg = torch.zeros((25, src.shape[1])).fill_(2).long().to(src.device)
            assert teacher_forcing_ratio == 0, 'techer forcing must be 0 during inference, i.e. use the y prediction'

        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device) #tensor to store decoder outputs
        context = self.encoder(src) #last hidden state of the encoder. same for all time step
        hidden = context #initial hidden state of the decoder is the last hidden state of encoder
        
        input = trg[0,:] #first input to the decoder is the <sos> token which is the first row in trg
        for t in range(1, max_len):
            output, hidden = self.decoder(input, hidden, context)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1] #the prediction of the next output
            input = (trg[t] if teacher_force else top1) #the next input for decoder is either the real y or the prediction
        return outputs

        

        
