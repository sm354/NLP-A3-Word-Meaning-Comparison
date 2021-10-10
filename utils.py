## Various utility functions
import os
import time
import datetime
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

def parse_args():
	parser = ArgumentParser(description='NLP A3-A')
	parser.add_argument('--dataset', '-d', type=str, default='data')
	parser.add_argument('--model', '-m', type=str, default='bilstm')
	parser.add_argument('--gpu', type=int, default=0)
	parser.add_argument('--batch_size', type=int, default=32)
	parser.add_argument('--hidden_dim', type=int, default=128)
	parser.add_argument('--epochs', type=int, default=15)
	parser.add_argument('--lr', type=float, default=0.001)
	return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
	# --result_dir
	check_folder(os.path.join(args.dataset))

	# --epoch
	try:
			assert args.epochs >= 1
	except:
			print('number of epochs must be larger than or equal to one')

	# --batch_size
	try:
			assert args.batch_size >= 1
	except:
			print('batch size must be larger than or equal to one')
	return args

def get_device(gpu_no):
	if torch.cuda.is_available():
		torch.cuda.set_device(gpu_no)
		return torch.device('cuda:{}'.format(gpu_no))
	else:
		return torch.device('cpu')

def makedirs(name):
	"""helper function for python 2 and 3 to call os.makedirs()
		avoiding an error if the directory to be created already exists"""

	import os, errno

	try:
		os.makedirs(name)
	except OSError as ex:
		if ex.errno == errno.EEXIST and os.path.isdir(name):
			# ignore existing directory
			pass
		else:
			# a different error happened
			raise

def check_folder(log_dir):
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)
	return log_dir

def get_logger(args, phase):
	logging.basicConfig(level=logging.INFO, 
												filename = "{}/{}_{}.log".format(args.dataset, args.model, phase),
												format = '%(asctime)s - %(message)s', 
												datefmt='%d-%b-%y %H:%M:%S')
	return logging.getLogger(phase)