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
from model import *
from utils import *

def main():
    hidden_dim=128
    device = get_device(0)

    x_test = pandas.read_csv("test.data.txt", delimiter='\t', header=None, names=['word', 'POS', 'position', 'sen1', 'sen2'])
    word1 = x_test.loc[:,'position'].apply(lambda positions : positions.split("-")[0])
    word2 = x_test.loc[:,'position'].apply(lambda positions : positions.split("-")[1])
    x_test = x_test.drop("position", axis=1)
    x_test.loc[:, "POS"] = x_test.loc[:, "POS"].apply(lambda POS_TAG : 0 if POS_TAG=="V" else 1)
    x_test.insert(3, "word1", word1)
    x_test.insert(5, "word2", word2)
    x_test.to_csv("data/test.csv", header=None, index=False)

    with open(os.path.join("data", "Field_TEXT"), 'rb') as f:
        TEXT = dill.load(f)

    fields = [
        ('word', TEXT),
        ('POS', data.Field(use_vocab=False, sequential=False)), 
        ('sen1', TEXT),
        ('word1', data.Field(use_vocab=False, sequential=False)),
        ('sen2', TEXT),
        ('word2', data.Field(use_vocab=False, sequential=False)),
    ]

    test_set = data.TabularDataset.splits(
        path = "data",
        test = "test.csv",
        format = 'csv',
        fields = fields,
        skip_header = False
    )

    (test_itr,) = data.BucketIterator.splits(
        test_set,
        #sort_key = lambda sample : len(sample.sen1),
        sort = False,
        batch_size = 32,
        repeat = False
    )

    model = myModel(pretrained_embeddings=TEXT.vocab.vectors, embed_dim=TEXT.vocab.vectors.shape[1], hidden_dim=hidden_dim, device=device)
    saved_model = torch.load("data/best-biLSTM-params.pt", map_location=device)
    print(saved_model["accuracy"])
    model.load_state_dict(saved_model['model_dict'])
    model = model.to(device)

    model.eval()
    test_itr.init_epoch()
    predictions = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_itr):
            answer = model(batch)
            answer = list(np.array((answer>=0.5).long().cpu()))
            predictions += answer

    predictions = pandas.DataFrame(predictions)[0].apply(lambda label : "T" if label == 1 else "F")
    predictions.to_csv("output.txt", header=None, index=None)
    print("predictions written in output.txt")

if __name__ == "__main__":
    parser = ArgumentParser(description='NLP A3-A')
    parser.add_argument('--score', action='store_true')
    args = parser.parse_args()

    if args.score:
        pred = np.array(pandas.read_csv("output.txt", header=None)[0])
        gold = np.array(pandas.read_csv("data/test.gold.txt", header=None)[0])

        print((pred==gold).mean())

    else:
        main()