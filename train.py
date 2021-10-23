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
# from torchtext.legacy import data
# from torchtext.data.utils import get_tokenizer

from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
from transformers import TrainingArguments

from model import *
from dataset import *
from utils import *

set_seed()

class Train():
    def __init__(self):
        print("program execution start: {}".format(datetime.datetime.now()))
        self.args = parse_args()
        self.device = get_device(self.args.gpu)
        self.logger = get_logger(self.args, "train")
        self.logger.info("Arguments: {}".format(self.args))

        self.trainset = myDataset(self.args.dataset, max_len=128, train=True, tokenizer='bert-base-uncased')
        self.devset = myDataset(self.args.dataset, max_len=128, train=False, tokenizer='bert-base-uncased')

        # self.trainloader = DataLoader(trainset, batch_size = args.batch_size, shuffle = True)
        # self.devloader = DataLoader(devset, batch_size = args.batch_size, shuffle = False)

        # self.model = myModel(self.device)
        self.model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

        # self.model = self.model.to(self.device)
        # self.criterion = nn.BCELoss()
        # self.opt = O.Adam(self.model.parameters(), lr = self.args.lr) #, weight_decay=0.005)
        # self.best_val_acc = None
        # self.scheduler = StepLR(self.opt, step_size=5, gamma=0.5)

        print("resource preparation done: {}".format(datetime.datetime.now()))

    def execute(self):
        training_args = TrainingArguments(
            output_dir="dir", overwrite_output_dir=True, 
            evaluation_strategy="epoch", learning_rate=5e-5,
            num_train_epochs=3, seed=4,
        )

        trainer = Trainer(
            model=self.model, args=training_args,
            train_dataset=self.trainset, eval_dataset=self.devset,
        )

        trainer.train()

if __name__ == "__main__":
    ## Start training
    task = Train()
    task.execute()