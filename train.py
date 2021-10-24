import os
import time
import random
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

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from transformers import AdamW, get_scheduler
from datasets import load_metric

from model import *
from dataset import *
from utils import *

set_seed()

class Train():
    def __init__(self):
        print("program execution start: {}".format(datetime.datetime.now()))
        self.args = parse_args()
        self.device = get_device(self.args.gpu)
        # self.logger = get_logger(self.args, "train")
        # self.logger.info("Arguments: {}".format(self.args))

        self.trainset = myDataset(self.args.dataset, max_len=100, train=True, model_name=self.args.model_name)
        self.devset = myDataset(self.args.dataset, max_len=100, train=False, model_name=self.args.model_name)

        self.trainloader = DataLoader(self.trainset, batch_size = self.args.batch_size, shuffle = True)
        self.devloader = DataLoader(self.devset, batch_size = self.args.batch_size, shuffle = False)

        self.model = myModel(self.device)
        self.model = self.model.to(self.device)

        self.criterion = nn.BCEWithLogitsLoss()
        self.best_val_acc = None
        self.optim = AdamW(self.model.parameters(), lr=5e-5, weight_decay=0.01)
        self.scheduler = get_scheduler("linear", self.optim, num_training_steps=self.args.epochs*len(self.trainloader), num_warmup_steps=500)

        print("resource preparation done: {}".format(datetime.datetime.now()))

    def train(self,epoch):
        self.model.train() 
        n_correct, n_total, n_loss = 0, 0, 0
        for batch_idx, batch in tqdm(enumerate(self.trainloader)):
            # if batch.batch_size != self.args.batch_size:
            #     print(batch.batch_size)
            self.optim.zero_grad()
            labels = batch['labels'].to(self.device)
            answer = self.model(batch)[:, 0]
            loss = self.criterion(answer, labels.float())
            answer = (answer>=0.5).long()
            # answer= answer.argmax(axis=-1).long()

            n_correct += (answer == labels).sum().item()
            n_total += labels.shape[0]
            n_loss += loss.item()

            loss.backward()
            # nn.utils.clip_grad_norm_(self.model.parameters(), 1, error_if_nonfinite=True)
            self.scheduler.step()
            self.optim.step()

        # if epoch == 20:
            # set_trace()
        train_loss = n_loss/n_total
        train_acc = 100. * n_correct/n_total
        return train_loss, train_acc
    
    def validate(self,epoch):
        self.model.eval() 
        n_correct, n_total, n_loss = 0, 0, 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.devloader):
                labels = batch['labels'].to(self.device)
                answer = self.model(batch)[:, 0]
                loss = self.criterion(answer, labels.float())
                answer = (answer>=0.5).long()
                # answer= answer.argmax(axis=-1).long()

                n_correct += (answer == labels).sum().item()
                n_total += labels.shape[0]
                n_loss += loss.item()

            val_loss = n_loss/n_total
            val_acc = 100. * n_correct/n_total
            return val_loss, val_acc

    def result_checkpoint(self, epoch, train_loss, val_loss, train_acc, val_acc, took):
        if self.best_val_acc is None or val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            torch.save({
                'accuracy': self.best_val_acc,
                'model_dict': self.model.state_dict(),
            }, '{}/best-{}-params.pt'.format(self.args.results_dir, self.args.model))
        # self.logger.info('| Epoch {:3d} | train loss {:5.2f} | train acc {:5.2f} | val loss {:5.2f} | val acc {:5.2f} | time: {:5.2f}s |'
                # .format(epoch, train_loss, train_acc, val_loss, val_acc, took))

    def execute(self):
        print(" [*] Training starts!")
        print('-' * 99)
        pbar = tqdm(range(1, self.args.epochs+1))
        for epoch in pbar:
            start = time.time()

            train_loss, train_acc = self.train(epoch)
            val_loss, val_acc = self.validate(epoch)

            took = time.time()-start
            # self.result_checkpoint(epoch, train_loss, val_loss, train_acc, val_acc, took)

            pbar.set_description('| Epoch {:3d} | train loss {:5.2f} | train acc {:5.2f} | val loss {:5.2f} | val acc {:5.2f} | time: {:5.2f}s |'.format(
                epoch, train_loss, train_acc, val_loss, val_acc, took))
        self.finish()

    def finish(self):
        # self.logger.info("[*] Training finished!\n\n")
        print('-' * 99)
        print(" [*] Training finished!")
        print("best validation accuracy:", self.best_val_acc)
    
if __name__ == "__main__":
    ## Start training
    task = Train()
    task.execute()