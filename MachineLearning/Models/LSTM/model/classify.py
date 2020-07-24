import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import numpy as np
import config

config = config.Config()

class att_cls(nn.Module):
    def __init__(self):
        super(att_cls, self).__init__()
        # text
        #
        self.v = nn.Parameter(torch.ones(config.hiden))
        # embed map num into a vector/tensor
        self.embed = nn.Embedding(config.vob_size, config.emb_size)
        # lstm
        self.lstm = nn.LSTM(config.emb_size, config.hiden, bidirectional=False, num_layers=1, batch_first=True)
        # map n-d into m-d space, get ready for attention
        self.fc_h = nn.Linear(config.hiden, config.att_size)
        self.fc_state = nn.Linear(config.hiden, config.att_size)
        self.fc_out = nn.Linear(config.att_size, 2)

        # title
        self.v_h = nn.Parameter(torch.ones(config.hiden))
        self.embed_h = nn.Embedding(config.vob_size, config.emb_size)
        self.lstm_h = nn.LSTM(config.emb_size, config.hiden, bidirectional=False, num_layers=1, batch_first=True)
        self.fc_h_h = nn.Linear(config.hiden, config.att_size)
        self.fc_state_h = nn.Linear(config.hiden, config.att_size)
        self.fc_out_h = nn.Linear(config.att_size, 2)

        # linearly weights
        self.weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, text, text_pad_mask, headline, headline_mask):
        #text
        # to 3-d
        x = self.embed(text)
        # self circle, states = h
        states, (h, c) = self.lstm(x)
        # map hidden size into attention size for last word
        h = self.fc_h(h.squeeze(0)).unsqueeze(1)
        # same for every word
        states = self.fc_state(states)
        # e = attention
        e = torch.sum(torch.mul(self.v, F.tanh(h + states)), dim=2)  # batch seqlen
        #mask pads into  np.inf
        att = F.softmax(e.masked_fill(text_pad_mask, -np.inf))
        #
        context = torch.matmul(att.unsqueeze(1), states).squeeze(1)

        #title
        x_h = self.embed_h(headline)
        states_h, (h, c) = self.lstm_h(x_h)
        h = self.fc_h_h(h.squeeze(0)).unsqueeze(1)
        states_h = self.fc_state(states_h)
        e_h = torch.sum(torch.mul(self.v, F.tanh(h + states_h)), dim=2)
        att_h = F.softmax(e_h.masked_fill(headline_mask, -np.inf))
        context_h = torch.matmul(att_h.unsqueeze(1), states_h).squeeze(1)


        output = self.fc_out(context)
        output_h = self.fc_out_h(context_h)

        result = self.weight * output + (1 - self.weight) * output_h

        return torch.softmax(result,-1)
