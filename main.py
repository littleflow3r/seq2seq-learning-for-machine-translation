import sys
import os
import math

import torch
import torch.optim as optim
import torch.nn as nn

import torchtext
from torchtext.data import Field, BucketIterator, TabularDataset

from model import Encoder, Decoder, Seq2Seq

import random
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.Load('../../spiece/ja.wiki.bpe.vs5000.model')

def tokenize_jp(x):
    x = str(x).lower()
    #if want to reverse the order of the sentences
    #return sp.EncodeAsPieces(x)[::-1]
    return sp.EncodeAsPieces(x)

def tokenize_en(x):
    x = str(x).lower()
    x = x.translate({ord(c): None for c in '!.?,'})
    return x.split()

SRC = Field(tokenize=tokenize_jp, init_token='<sos>', eos_token='<eos>')
TRG = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>')

dataset = TabularDataset(path='data/csvfile.csv', format='csv', fields=[('id', None),('jp', SRC), ('en', TRG), ('cn', None)], skip_header=True)
train_dt, valid_dt, test_dt = dataset.split(split_ratio=[0.7, 0.1, 0.2], random_state=random.getstate())

#print (len(train.examples), len(valid.examples), len(test.examples))
#print (vars(train.examples[0]))

SRC.build_vocab(train_dt, min_freq=2)
TRG.build_vocab(train_dt, min_freq=2)
#print (len(SRC.vocab), len(TRG.vocab))
#print (SRC.vocab.freqs.most_common(10))
#print (TRG.vocab.freqs.most_common(10))
#print ('index 3 in the trg:', TRG.vocab.itos[3])
#print ('index the in the trg:', TRG.vocab.stoi['the'])

bsize = 32
gpu = False
device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')
train_it, valid_it, test_it = BucketIterator.splits((train_dt, valid_dt, test_dt), batch_size=bsize, sort_key=lambda x: len(x.jp), sort_within_batch=False, device=device)

'''
for b in train_it:
    print (b.jp, b.en)
    sys.exit()
'''

def train(model, train_it, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(train_it):
        src = batch.jp
        trg = batch.en
        optimizer.zero_grad()
        output = model(src, trg)
        loss = criterion(output[1:].view(-1, output.shape[2]), trg[1:].view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss/ len(train_it)

def evaluate(model, data_it, criterion):
    model.eval()
    epoch_loss = 0
    for i, batch in enumerate(data_it):
        src = batch.jp
        trg = batch.en
        output = model(src, trg, 0)
        loss = criterion(output[1:].view(-1, output.shape[2]), trg[1:].view(-1))
        epoch_loss += loss.item()
    return epoch_loss/ len(data_it)

input_dim = len(SRC.vocab)
out_dim = len(TRG.vocab)
enc_emb_dim = 128
dec_emb_dim = 128
hidden_dim = 256
nlayers = 2
enc_dropout = 0.3
dec_dropout = 0.3
enc = Encoder(input_dim, enc_emb_dim, hidden_dim, nlayers, enc_dropout)
dec = Decoder(out_dim, dec_emb_dim, hidden_dim, nlayers, dec_dropout)
model = Seq2Seq(enc, dec, device).to(device)

optimizer = optim.Adam(model.parameters())
pad_idx = TRG.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

epoch = 10
clip = 10
savedir = 'models'
model_save_path = os.path.join(savedir, 's2smodel.pt')
best_valid_loss = float('inf')

if not os.path.isdir(f'{savedir}'):
    os.makedirs(f'{savedir}')
for ep in range(epoch):
    train_loss = train(model, train_it, optimizer, criterion, clip)
    valid_loss = evaluate(model, valid_it, criterion)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), model_save_path)
    
    print (f'epoch: {ep+1:03} | train loss: {train_loss: .3f} | train_ppl: {math.exp(train_loss):7.3f} | Val. Loss: {valid_loss:.3f} | Val. PPL: {math.exp(valid_loss):7.3f} |')

model.load_state_dict(torch.load(model_save_path))
test_loss = evaluate(model, test_it, criterion)
print(f'|test loss: {test_loss: .3f} | test_ppl: {math.exp(test_loss):7.3f}|')

def translate_sentence(sentence):
    tokenized = tokenize_jp(sentence)
    numericalised = [SRC.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(numericalised).unsqueeze(1).to(device)
    translation_tensor_probs = model(tensor, None, 0).squeeze(1)
    translation_tensor = torch.argmax(translation_tensor_probs, 1)
    translation = [TRG.vocab.itos[t] for t in translation_tensor][1:]
    return translation

candidate = ' '.join(vars(valid_dt.examples[2])['jp'])
#candidate = '私はリンゴが好きです'
candidate_translation = ' '.join(vars(valid_dt.examples[2])['en'])
print (candidate)
print (candidate_translation)
print (translate_sentence(candidate))
tokenized = tokenize_jp(candidate)
numericalised = [SRC.vocab.stoi[t] for t in tokenized]
back_to_candidate = [SRC.vocab.itos[n] for n in numericalised][1:]
print (back_to_candidate)
