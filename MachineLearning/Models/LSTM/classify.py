import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import numpy as np
from lconfig import Config

config = Config()

class att_cls(nn.Module):
    def __init__(self):
        super(att_cls, self).__init__()
        # text
        #
        self.v = Variable(torch.ones(config.hiden)).cuda() if config.gpu else Variable(torch.ones(config.hiden))
        # embed map num into a vector/tensor
        self.embed = nn.Embedding(config.vob_size, config.emb_size)
        # lstm
        self.lstm = nn.LSTM(config.emb_size, config.hiden, bidirectional=False, num_layers=1, batch_first=True)
        self.w = nn.Parameter(torch.zeros(config.hiden))
        # map n-d into m-d space, get ready for attention
        self.fc_out = nn.Linear(config.hiden, 2)

        # title
        self.v_h = Variable(torch.ones(config.hiden)).cuda() if config.gpu else Variable(torch.ones(config.hiden))
        self.embed_h = nn.Embedding(config.vob_size, config.emb_size)
        self.lstm_h = nn.LSTM(config.emb_size, config.hiden, bidirectional=False, num_layers=1, batch_first=True)
        self.w_h = nn.Parameter(torch.zeros(config.hiden))
        self.fc_out_h = nn.Linear(config.hiden, 2)

        # linearly weights
        self.weight = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, text, text_pad_mask, headline, headline_mask):
        #text
        # to 3-d
        x = self.embed(text)
        # self circle, states = h
        states,_= self.lstm(x)

        att = F.softmax(torch.matmul(states, self.w), dim=1).unsqueeze(1)

        output=self.fc_out(torch.matmul(att,states).squeeze())


        #title
        x_h = self.embed_h(headline)
        states_h, _ = self.lstm_h(x_h)
       
        att_h = F.softmax(torch.matmul(states_h, self.w_h), dim=1).unsqueeze(1)
        output_h = self.fc_out(torch.matmul(att_h,states_h).squeeze())

        result = self.weight * output + (1 - self.weight) * output_h

        return torch.softmax(result,-1)
