import os
import copy
import time
import datetime
from typing import Text
import pandas
import dill
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

def process_data(path, train=True):
    train = "train" if train else "validation"
    x_train = pandas.read_csv(os.path.join(path, "%s/%s.data.txt"%(train, train)), delimiter='\t', header=None, names=['word', 'POS', 'position', 'sen1', 'sen2'])
    y_train = pandas.read_csv(os.path.join(path, "%s/%s.gold.txt"%(train, train)), header=None, names=['label'])
    dataset = pandas.concat((x_train, y_train), axis=1)
    word1 = dataset.loc[:,'position'].apply(lambda positions : positions.split("-")[0])
    word2 = dataset.loc[:,'position'].apply(lambda positions : positions.split("-")[1])
    dataset = dataset.drop("position", axis=1)
    dataset.loc[:, "POS"] = dataset.loc[:, "POS"].apply(lambda POS_TAG : 0 if POS_TAG=="V" else 1)
    dataset.loc[:, "label"] = dataset.loc[:, "label"].apply(lambda lab : 0 if lab=="F" else 1)
    dataset.insert(3, "word1", word1)
    dataset.insert(5, "word2", word2)
    dataset.to_csv(os.path.join(path, "%s/%s.csv"%(train, train)), header=None, index=False)

class myDataset:
    def __init__(self, args):
        self.args = args
        
        process_data(args.dataset, train=True)
        process_data(args.dataset, train=False)

        TEXT = data.Field(
            sequential=True,
            lower=True,
            tokenize=lambda sen : sen.split(" "),
        )
        # get_tokenizer("basic_english"),
        fields = [
            ('word', TEXT),
            ('POS', data.Field(use_vocab=False, sequential=False)), 
            ('sen1', TEXT),
            ('word1', data.Field(use_vocab=False, sequential=False)),
            ('sen2', TEXT),
            ('word2', data.Field(use_vocab=False, sequential=False)),
            ('label', data.Field(use_vocab=False, sequential=False)),
        ]

        train_set, val_set = data.TabularDataset.splits(
            path = args.dataset,
            train = 'train/train.csv',
            validation = 'validation/validation.csv',
            format = 'csv',
            fields = fields,
            skip_header = False
        )

        TEXT.build_vocab(train_set, vectors='glove.6B.100d')

        train_itr, val_itr = data.BucketIterator.splits(
            (train_set, val_set),
            #sort_key = lambda sample : len(sample.sen1),
            sort = False,
            batch_size = 32,
            repeat = False
        )
        with open(os.path.join(self.args.dataset, "Field_TEXT"), 'wb') as f:
            dill.dump(TEXT, f)
        print("TEXT Field saved in %s"%self.args.dataset)
        
        self.vocab = TEXT.vocab
        self.train_iter = train_itr
        self.dev_iter = val_itr