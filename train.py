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

        self.trainset = WiC_dataset(args.dataset, max_len=128, train=True, tokenizer='bert-base-uncased')
        self.devset = WiC_dataset(args.dataset, max_len=128, train=False, tokenizer='bert-base-uncased')

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

    def train(self,epoch):
        self.model.train() #; self.dataset.train_iter.init_epoch()
        n_correct, n_total, n_loss = 0, 0, 0
        for batch_idx, batch in enumerate(self.trainloader):
            # if batch.batch_size != self.args.batch_size:
            #     print(batch.batch_size)
            self.opt.zero_grad()
            batch.label = batch.label.to(self.device)
            answer = self.model(batch)
            loss = self.criterion(answer, batch.label.float())
            answer = (answer>=0.5).long()

            n_correct += (answer == batch.label).sum().item()
            n_total += batch.batch_size
            n_loss += loss.item()

            loss.backward()
            # nn.utils.clip_grad_norm_(self.model.parameters(), 1, error_if_nonfinite=True)
            self.opt.step()
        # if epoch == 20:
            # set_trace()
        train_loss = n_loss/n_total
        train_acc = 100. * n_correct/n_total
        return train_loss, train_acc

    def validate(self,epoch):
        self.model.eval() #; self.dataset.dev_iter.init_epoch()
        n_correct, n_total, n_loss = 0, 0, 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.devloader):
                batch.label = batch.label.to(self.device)
                answer = self.model(batch)
                loss = self.criterion(answer, batch.label.float())
                answer = (answer>=0.5).long()

                n_correct += (answer == batch.label).sum().item()
                n_total += batch.batch_size
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
        self.logger.info('| Epoch {:3d} | train loss {:5.2f} | train acc {:5.2f} | val loss {:5.2f} | val acc {:5.2f} | time: {:5.2f}s |'
                .format(epoch, train_loss, train_acc, val_loss, val_acc, took))

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


        # print(" [*] Training starts!")
        # print('-' * 99)
        # pbar = tqdm(range(1, self.args.epochs+1))
        # for epoch in pbar:
        #     start = time.time()

        #     train_loss, train_acc = self.train(epoch)
        #     val_loss, val_acc = self.validate(epoch)
        #     # self.scheduler.step()

        #     took = time.time()-start
        #     self.result_checkpoint(epoch, train_loss, val_loss, train_acc, val_acc, took)

        #     pbar.set_description('| Epoch {:3d} | train loss {:5.2f} | train acc {:5.2f} | val loss {:5.2f} | val acc {:5.2f} | time: {:5.2f}s |'.format(
        #         epoch, train_loss, train_acc, val_loss, val_acc, took))
        # self.finish()

    def finish(self):
        self.logger.info("[*] Training finished!\n\n")
        print('-' * 99)
        print(" [*] Training finished!")
        print("best validation accuracy:", self.best_val_acc)

if __name__ == "__main__":
    ## Start training
    task = Train()
    task.execute()