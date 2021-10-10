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
from dataset import *
from utils import *

# fixing seeds to reproduce results (exact!)
torch.manual_seed(4)
random.seed(4)
np.random.seed(4)

class Train():
    def __init__(self):
        print("program execution start: {}".format(datetime.datetime.now()))
        self.args = parse_args()
        self.device = get_device(self.args.gpu)
        self.logger = get_logger(self.args, "train")
        self.logger.info("Arguments: {}".format(self.args))

        dataset_options = {
            'batch_size': self.args.batch_size, 
            'device': self.device
            }

        self.dataset = myDataset(self.args)
        self.vocab = self.dataset.vocab
        self.embed_dim = self.vocab.vectors.shape[1]

        self.model = myModel(self.vocab.vectors, self.embed_dim, self.args.hidden_dim, self.device)

        self.model = self.model.to(self.device)
        self.criterion = nn.BCELoss()
        self.opt = O.Adam(self.model.parameters(), lr = self.args.lr)
        self.best_val_acc = None
        # self.scheduler = StepLR(self.opt, step_size=5, gamma=0.5)

        print("resource preparation done: {}".format(datetime.datetime.now()))

    def train(self):
        self.model.train(); self.dataset.train_iter.init_epoch()
        n_correct, n_total, n_loss = 0, 0, 0
        for batch_idx, batch in enumerate(self.dataset.train_iter):
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
            # nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.opt.step()
        train_loss = n_loss/n_total
        train_acc = 100. * n_correct/n_total
        return train_loss, train_acc

    def validate(self):
        self.model.eval(); self.dataset.dev_iter.init_epoch()
        n_correct, n_total, n_loss = 0, 0, 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.dataset.dev_iter):
                batch.label = batch.label.to(self.device)
                answer = self.model(batch)
                loss = self.criterion(answer, batch.label.float())
                answer = (answer>=0.5).long()

                n_correct += (answer == batch.label).sum().item()
                n_total += batch.batch_size
                n_loss += loss.item()

            # print(np.bincount(answer.cpu()), np.bincount(batch.label.cpu()))
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
        print(" [*] Training starts!")
        print('-' * 99)
        pbar = tqdm(range(1, self.args.epochs+1))
        for epoch in pbar:
            start = time.time()

            train_loss, train_acc = self.train()
            val_loss, val_acc = self.validate()
            # self.scheduler.step()

            took = time.time()-start
            self.result_checkpoint(epoch, train_loss, val_loss, train_acc, val_acc, took)

            pbar.set_description('| Epoch {:3d} | train loss {:5.2f} | train acc {:5.2f} | val loss {:5.2f} | val acc {:5.2f} | time: {:5.2f}s |'.format(
                epoch, train_loss, train_acc, val_loss, val_acc, took))
        self.finish()

    def finish(self):
        self.logger.info("[*] Training finished!\n\n")
        print('-' * 99)
        print(" [*] Training finished!")
        print("best validation accuracy:", self.best_val_acc)

if __name__ == "__main__":
    ## Start training
    task = Train()
    task.execute()