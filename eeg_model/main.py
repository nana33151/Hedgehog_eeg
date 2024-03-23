import torch
from torch import nn
import math
import mne
from mne import Epochs
from mne import io
from mne.datasets import eegbci
from mne import channels
import numpy as np
import torch.utils.data as data
import torch.nn.functional as F
from torch import optim
import time
import random
import braindecode

from braindecode.datasets import MOABBDataset, BaseConcatDataset
from numpy import multiply
from braindecode.preprocessing import create_windows_from_events

from braindecode.preprocessing import (
    Preprocessor,
    exponential_moving_standardize,
    preprocess,
)

TGT_VOCAB_SIZE = 5
DIM=64
NUM_HEADS=4
NUM_LAYERS=4
FF_DIM = 64
DROPOUT = 0.5
N_CHANNELS = 22
SEQ_LEN = 1000

class PositionalEncoding(nn.Module):
    def __init__(self, seq_length, model_dim):
        super(PositionalEncoding,self).__init__()
        self.sl = seq_length
        self.md = model_dim
        self.encodings_matrices = torch.zeros(self.sl, self.md).cuda()
        position = torch.arange(0,self.sl,dtype = torch.float).unsqueeze(1).cuda()
        for p in range(self.sl):
          for i in range(self.md):
            t = p/(10000**(2*i/self.md))
            if i%2 == 0:
                self.encodings_matrices[p,i] = math.cos(t)
            else:
                self.encodings_matrices[p,i] = math.sin(t)

    def forward(self,x):
      output = torch.cat((x,self.encodings_matrices),1).cuda()
      return output

class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_ff, seq_lenght, dropout,in_d,tgt_vocab_size):
        super(Transformer, self).__init__()

        self.in_d = in_d
        self.seq_lenght = seq_lenght
        self.encoder_embedding = nn.Linear(in_d,d_model)#!
        self.decoder_embedding = nn.Linear(in_d,d_model)#!
        self.positional_encoding = PositionalEncoding(seq_lenght, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model*2, num_heads, d_ff, dropout)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model*2, num_heads, d_ff, dropout)

        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers)
        self.fc = nn.Linear(d_model*2, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
        self.softmax = nn.Softmax(dim = 1)
        self.mask = torch.triu(torch.ones(seq_lenght,d_model*2), diagonal=1)
        self.mask = self.mask.int().float().cuda()

    def forward(self,src,tgt, valid = False):
        if valid:
            tgt_embedded = tgt.cuda()
        else:
            tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt.transpose(0,1)))).cuda()
            tgt_embedded = tgt_embedded * self.mask
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src.transpose(0,1)))).cuda()
        enc_output = self.encoder.forward(src_embedded).cuda()
        dec_output = self.decoder.forward(tgt_embedded, enc_output).cuda()
        fc_output = self.fc(dec_output).cuda()
        output = self.softmax(fc_output).cuda()
        return [output, src_embedded]

    def accuracy(self, output, labels):
        correct = 0
        all_points = output.size(0)
        for i in range(all_points):
            pred = torch.argmax(output[i]).item()
            true_pred = torch.argmax(labels[i]).item()
            if pred == true_pred:
                correct += 1
        return correct

    def training_step(self, batch):
        data, test, lab = batch
        self.optimizer.zero_grad()
        output, _ = self.forward(data, test)
        f_output = output.view(-1, TGT_VOCAB_SIZE)
        loss = self.loss(f_output,lab)
        loss.backward()
        self.optimizer.step()
        correct = self.accuracy(f_output,lab)
        return [loss, correct]

    def valid_step(self,batch):
        data,y,true_pred = batch
        output, embeddings = self.forward(data, y, valid = True)
        f_output = output.view(-1,TGT_VOCAB_SIZE)
        loss = self.loss(f_output, true_pred)
        correct = self.accuracy(f_output, true_pred)
        return [loss,correct,embeddings]


def labels_to_matrices(labels, tgt_vocab_size, seq_len):
    class_matrices = torch.zeros(seq_len, tgt_vocab_size-1).cuda()
    last_class = torch.ones(seq_len, 1).cuda()
    labels_matrices = torch.cat([class_matrices, last_class],1).cuda()
    coordinates = list(labels.keys())
    for i in coordinates:
        labels_matrices[i][labels[i]-1] = 1.0
        labels_matrices[i][tgt_vocab_size-1] = 0.0
    labels_matrices = labels_matrices.type(torch.FloatTensor).cuda()
    return(labels_matrices)

def slice_to_batches(raw_data, batch_size, n_batches, n_chans):
  batch_list = []
  for b in range(n_batches):
    single_batch = []
    for i in range(n_chans):
      element = raw_data[i][(b*batch_size):((b+1)*batch_size)]
      element = element.unsqueeze(0).cuda()
      single_batch.append(element)
    tensored = torch.cat(single_batch,0).type(torch.FloatTensor).cuda()
    batch_list.append(tensored)
  return batch_list

def preprocessing(dataset):

    raw_channels = dataset.datasets[0].raw.info['chs']
    N_CHANNELS = len(raw_channels)-4

    low_cut_hz = 4.
    high_cut_hz = 38.
    factor_new = 1e-3
    init_block_size = 1000
    factor = 1e6

    preprocessors = [
        Preprocessor('pick_types', eeg=True, meg=False, stim=False),
        Preprocessor(lambda data: multiply(data, factor)),
        Preprocessor('filter', l_freq=low_cut_hz, h_freq=high_cut_hz),
        Preprocessor(exponential_moving_standardize,
                    factor_new=factor_new, init_block_size=init_block_size)
    ]

    preprocess(dataset, preprocessors, n_jobs=1)

    return dataset

one_train_set = 4

training_set = []
target_set = []
validating_set = []
for id in range(1,10):
    raw_dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[id])
    preprocessed_dataset = preprocessing(raw_dataset)
    training_set += preprocessed_dataset.datasets[0:4]
    target_set += preprocessed_dataset.datasets[4:8]
    validating_set += preprocessed_dataset.datasets[8:12]

training_datasets = []
test_datasets = []
labels_batches = []
validating_datasets = []
pred_batches = []

for i in range(len(validating_set)):
    valid_raw = validating_set[i].raw
    raw_data = torch.from_numpy(valid_raw.get_data()).cuda()
    print(raw_data.shape)
    n_batches = raw_data.size(1)//SEQ_LEN
    validating_datasets += slice_to_batches(raw_data, SEQ_LEN, n_batches, N_CHANNELS)
    true_preds = torch.from_numpy(mne.events_from_annotations(valid_raw)[0]).cuda()
    pred_dict = {}
    for l in true_preds:
        pred_dict[l[0].item()] = l[2].item()
    pred_matrices = labels_to_matrices(pred_dict, TGT_VOCAB_SIZE, n_batches * SEQ_LEN)
    pred_batches += torch.split(pred_matrices, SEQ_LEN)

for i in range(len(training_datasets)):
    train_raw = training_set[i].raw
    raw_data = torch.from_numpy(train_raw.get_data()).cuda()
    n_batches = raw_data.size(1)//SEQ_LEN
    training_datasets += slice_to_batches(raw_data, SEQ_LEN, n_batches, N_CHANNELS)
    test_raw = target_set[i].raw
    traw_data = torch.from_numpy(test_raw.get_data()).cuda()
    labels = torch.from_numpy(mne.events_from_annotations(test_raw)[0]).cuda()
    labels_dict = {}
    for l in labels:
        labels_dict[l[0].item()] = l[2].item()
    labels_matrices = labels_to_matrices(labels_dict, TGT_VOCAB_SIZE, n_batches * SEQ_LEN)
    labels_batches += torch.split(labels_matrices, SEQ_LEN)
    
    test_datasets += slice_to_batches(traw_data, SEQ_LEN, n_batches, N_CHANNELS)


transformer = Transformer(DIM,NUM_HEADS,NUM_LAYERS,FF_DIM,SEQ_LEN,DROPOUT,N_CHANNELS,TGT_VOCAB_SIZE)#d_model, num_heads, num_layers, d_ff, seq_lenght, dropout,in_d,tgt_vocab_size
transformer = transformer.cuda()
torch.save(transformer, "model.onnx")

running_loss = 0
last_loss = 0
running_corr = 0
EPOCHS = 1
output = 0
start_time = time.time()
best_loss = float('inf')
for j in range(EPOCHS):
    for i in range(len(training_datasets)):
        transformer.train()
        loss, corr = transformer.training_step([training_datasets[i],test_datasets[i],labels_batches[i]])
        running_loss += loss.item()
        running_corr += corr
        if loss < best_loss:
            best_loss = loss
            torch.save(transformer, "model.onnx")
        if i % 10 == 9:
            last_loss = running_loss / 10 
            
            print('  batch {} loss: {} correct {}'.format(i + 1, last_loss, running_corr/10000))
            running_loss = 0
            running_corr = 0

embeddings = torch.randn(SEQ_LEN,DIM*2)
valid_loss = 0
last_loss = 0
valid_corr = 0

for i in range(len(validating_datasets)):
    loss,corr,embeddings = transformer.valid_step([validating_datasets[i],embeddings, pred_batches[i]])
    valid_loss += loss.item()
    valid_corr += corr
    if i % 10 == 9:
        last_loss = running_loss / 10 
        print('  batch {} loss: {} correct {}'.format(i + 1, last_loss, running_corr/10000))
        valid_loss = 0
        valid_corr = 0
