import torch
seq_len = 20
tgt_vocab_size = 5
class_matrices = torch.zeros(seq_len, tgt_vocab_size-1)
last_class = torch.ones(seq_len, 1)
labels_matrices = torch.cat([class_matrices, last_class],1)
print(labels_matrices.shape)