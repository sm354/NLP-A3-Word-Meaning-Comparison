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

from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
from transformers import TrainingArguments
from datasets import load_metric

from model import *
from dataset import *
from utils import *

set_seed()

metric = load_metric("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

class Train():
    def __init__(self):
        print("program execution start: {}".format(datetime.datetime.now()))
        self.args = parse_args()
        self.device = get_device(self.args.gpu)
        # self.logger = get_logger(self.args, "train")
        # self.logger.info("Arguments: {}".format(self.args))

        self.trainset = myDataset(self.args.dataset, max_len=100, train=True, model_name=self.args.model_name)
        self.devset = myDataset(self.args.dataset, max_len=100, train=False, model_name=self.args.model_name)

        # self.trainloader = DataLoader(trainset, batch_size = args.batch_size, shuffle = True)
        # self.devloader = DataLoader(devset, batch_size = args.batch_size, shuffle = False)

        # self.model = myModel(self.device)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.args.model_name, num_labels=2)

        # self.model = self.model.to(self.device)
        # self.criterion = nn.BCELoss()
        # self.opt = O.Adam(self.model.parameters(), lr = self.args.lr) #, weight_decay=0.005)
        # self.best_val_acc = None
        # self.scheduler = StepLR(self.opt, step_size=5, gamma=0.5)

        print("resource preparation done: {}".format(datetime.datetime.now()))

    def execute(self):
        training_args = TrainingArguments(
            output_dir= self.args.results_dir, 
            overwrite_output_dir=True, 
            evaluation_strategy="epoch", 
            learning_rate=5e-5,
            weight_decay=0.01,
            num_train_epochs=self.args.epochs,
            seed=4,
            per_device_eval_batch_size=self.args.batch_size,
            per_device_train_batch_size=self.args.batch_size,
            warmup_steps=500,
            logging_dir='./logs',
            logging_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
        )

        trainer = Trainer(
            model=self.model, args=training_args,
            train_dataset=self.trainset, eval_dataset=self.devset,
            compute_metrics=compute_metrics# deplag
        )

        trainer.train()

if __name__ == "__main__":
    ## Start training
    task = Train()
    task.execute()