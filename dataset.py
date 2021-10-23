import os
# import spacy
from tqdm import tqdm
import numpy as np
import pandas
from pdb import set_trace
import torch

from torch.utils.data import Dataset
from transformers import AutoTokenizer

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
    # dataset.to_csv(os.path.join(path, "%s/%s.csv"%(train, train)), header=None, index=False)
    return dataset

class myDataset(Dataset):
    def __init__(self, data_dir, max_len=128, train=True, tokenizer='bert-base-uncased'):
        self.data_dir = data_dir
        self.max_len = max_len
        self.train = train

        # convert raw data into pandas data frame
        self.dataset_df = process_data(data_dir, train=train)

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, do_lower_case=True)

    def __getitem__(self, index):
        sample = self.dataset_df.iloc[index]

        x = self.tokenizer(sample['sen1'], sample['sen2'], return_tensors='pt', \
            add_special_tokens=True, max_length=self.max_len, padding='max_length')
        y = torch.tensor(sample['label'])

        # format as required by Train
        item = {k: v[0] for k,v in x.items()} # only one datapoint; shape: 128
        item['labels'] = y
        return item
    
    def __len__(self):
        return len(self.dataset_df)