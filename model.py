import os
import time
import random
import spacy
import dill
from tqdm import tqdm
import datetime
import numpy as np
import pandas
import logging
from argparse import ArgumentParser
from pdb import set_trace
import torch
import torch.optim as O
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torchtext.legacy import data
from torchtext.data.utils import get_tokenizer

class myModel(nn.Module):
    def __init__(self, pretrained_embeddings, embed_dim=100, hidden_dim=100, device=torch.device('cpu')):
        super(myModel, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.device = device
        
        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=True)
        # self.dropout = nn.Dropout(p = 0.5)
        self.bilstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=False, batch_first=True, num_layers=2)
        # self.fc = nn.Linear(hidden_dim, 2)
        # self.attn = nn.MultiheadAttention(embed_dim=300, num_heads=4, batch_first=True) #self.hidden_dim
        self.similarity = nn.CosineSimilarity(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        '''
            [torchtext.legacy.data.batch.Batch of size 32]
                [.word]:[torch.LongTensor of size 1x32]
                [.POS]:[torch.LongTensor of size 32]
                [.sen1]:[torch.LongTensor of size 17x32]
                [.word1]:[torch.LongTensor of size 32]
                [.sen2]:[torch.LongTensor of size 26x32]
                [.word2]:[torch.LongTensor of size 32]
                [.label]:[torch.LongTensor of size 32]
        '''
        input1 = input.sen1.permute(1,0).to(self.device)
        input2 = input.sen2.permute(1,0).to(self.device)

        # embed the tokens (indices in the vocab) 
        # out1 = self.dropout(self.embedding(input1))
        # out2 = self.dropout(self.embedding(input2))
        out1 = self.embedding(input1)
        out2 = self.embedding(input2)
        
        out1, (_, _) = self.bilstm(out1)
        out2, (_, _) = self.bilstm(out2)
        
        # mask1 = (input1 == 1)
        # mask2 = (input2 == 1)

        # attention box
        # (out1, _) = self.attn(out1, out1, out1, key_padding_mask=mask1)
        # (out2, _) = self.attn(out2, out2, out2, key_padding_mask=mask2)
        
        # word of interest
        woi1_idx = torch.repeat_interleave(torch.unsqueeze(torch.unsqueeze(input.word1, 1), 2), self.hidden_dim, dim=2).to(self.device)
        woi2_idx = torch.repeat_interleave(torch.unsqueeze(torch.unsqueeze(input.word2, 1), 2), self.hidden_dim, dim=2).to(self.device)
        
        # gather the word of interest for each sentence in the batch
        out1 = torch.gather(input=out1, dim=1, index=woi1_idx).squeeze()
        out2 = torch.gather(input=out2, dim=1, index=woi2_idx).squeeze()
        
        # linear layer
        #out1 = self.fc(out1)
        #out2 = self.fc(out2)
        
        # compute scores of similarity
        out = self.similarity(out1, out2)
        out = self.sigmoid(out)
        # try:
        #     assert out.shape[0] == self.bs
        # except:
        #     print("AssertionError", out.shape[0], self.bs) # all samples are touched in a epoch

        return out