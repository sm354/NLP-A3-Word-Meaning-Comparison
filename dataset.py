import os
import random
# import spacy
import dill
from tqdm import tqdm
import numpy as np
import pandas
from argparse import ArgumentParser
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
    dataset.to_csv(os.path.join(path, "%s/%s.csv"%(train, train)), header=None, index=False)
    return dataset

class WiC_dataset(Dataset):
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

        item = {}
        for k,v in x.items():
            item[k] = v[0] # only one datapoint; shape: 128
        item['labels'] = y

        return item
    
    def __len__(self):
        return len(self.dataset_df)

# en = spacy.load('en_core_web_sm')
# def tokeni(sen):
#     t = en.tokenizer(sen)
#     return [word.text for word in t]

# class myDataset:
#     def __init__(self, args, tokenizer='bert-base-uncased'):
#         self.args = args
        
#         process_data(args.dataset, train=True)
#         process_data(args.dataset, train=False)

#         # now we will encode each sentence into ids using tokenizer (of particular model)
#         # ie sentence -> tokens -> ids
#         # Also, special tokens will be added in the field

#         tokenizer = AutoTokenizer.from_pretrained(tokenizer) #, do_lower_case=True) handled in data.Field
#         spl_token_ids = {
#             "cls": tokenizer.cls_token_id, 
#             "sep": tokenizer.sep_token_id, 
#             "pad": tokenizer.pad_token_id, 
#             "mask": tokenizer.mask_token_id),
#         }

#         TEXT = data.Field(
#             sequential=True,
#             lower=True,
#             use_vocab=False,
#             tokenize=lambda sen : tokenizer.encode_plus(sen, add_special_tokens=False, return_tensors='pt'), 
#         )
#         # get_tokenizer("basic_english"),
#         fields = [
#             ('word', TEXT),
#             ('POS', data.Field(use_vocab=False, sequential=False)), 
#             ('sen1', TEXT),
#             ('word1', data.Field(use_vocab=False, sequential=False)),
#             ('sen2', TEXT),
#             ('word2', data.Field(use_vocab=False, sequential=False)),
#             ('label', data.Field(use_vocab=False, sequential=False)),
#         ]

#         train_set, val_set = data.TabularDataset.splits(
#             path = args.dataset,
#             train = 'train/train.csv',
#             validation = 'validation/validation.csv',
#             format = 'csv',
#             fields = fields,
#             skip_header = False,
#         )

#         # TEXT.build_vocab(train_set, vectors='glove.6B.300d')

#         train_itr, val_itr = data.BucketIterator.splits(
#             (train_set, val_set),
#             #sort_key = lambda sample : len(sample.sen1),
#             sort = False,
#             batch_size = args.batch_size,
#         )

#         # with open(os.path.join(self.args.results_dir, "Field_TEXT"), 'wb') as f:
#             # dill.dump(TEXT, f)
#         # print("TEXT Field saved in %s"%self.args.results_dir)
        
#         self.vocab = TEXT.vocab
#         self.train_iter = train_itr
#         self.dev_iter = val_itr
#         self.TEXT = TEXT
#         self.train_set = train_set
#         self.val_set = val_set
