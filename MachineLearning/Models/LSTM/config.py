import os
from functools import reduce
import torch
import glob
import math
import numpy as np


class Config(object):

    def __init__(self):


        self.hiden = 300
        self.n_layers = 1
        self.emb_size =300
        self.bidirectional = True
        self.title_max_seqlen = 50
        self.max_seqlen=200
        self.att_size=300
        self.lr=1e-3
        self.epoch=20
        self.batch_size=128

        self.log_step = 10
        self.sample_step = 100

        self.max_acc=0

        self.train_path='./data/train.csv'
        self.vocab_path='./data/vocab.txt'
        self.vob_w2id,self.vob_id2w,self.vob_size=self.W2id()

        self.gpu=torch.cuda.is_available()


    def W2id(self):
        with open(self.vocab_path, encoding='utf-8') as f:
            i = 0
            w2id, id2w = {}, {}
            for word in f:
                word = word.strip()
                w2id[word] = i
                id2w[i] = word
                i += 1
            return w2id, id2w, i






