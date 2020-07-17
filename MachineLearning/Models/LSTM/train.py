"""
Train classifier
"""

import torch.optim as optim
from model.classify import att_cls
from dataprocess import textDataset
from config import Config
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from config import Config
import os
import numpy as np

config = Config()


def train(model, train_loader, val_loader, opt):
    step = 0
    for epoch in range(1, 1 + config.epoch):
        for title, text, label, in train_loader:
            if config.gpu:
                title, text, label = title.cuda(), text.cuda(), label.cuda()

            opt.zero_grad()
            predict = model(text, text.eq(0), title, title.eq(0))
            loss = F.cross_entropy(predict, label)
            #print(predict, label)
            #print(loss)
            loss.backward()
            opt.step()
            step += 1
            if step % config.log_step == 0:
                correct = (torch.max(predict, 1)[1] == label).sum()
                acc = 100 * float(correct) / float(config.batch_size)
                print('\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(step,
                                                                               loss,
                                                                               acc,
                                                                               correct,
                                                                               config.batch_size))
            if step % config.sample_step == 0:
                val(model, val_loader)


def val(model, loader):
    model = model.eval()
    correct = 0
    for title, text, label, in loader:
        if config.gpu:
            title, text, label = title.cuda(), text.cuda(), label.cuda()

        predict = model(text, text.eq(0), title, title.eq(0))
        correct += (torch.max(predict, 1)[1] == label).sum()
    print(label)
    acc = 100 * float(correct) / float(len(loader.dataset))
    print('\r acc: {:.4f}%({}/{}) '.format(acc,
                                           correct,
                                           len(loader.dataset)))
    if acc > config.max_acc:
        config.max_acc = acc
        print('the model is saved and the acc is {}'.format(acc))
        if os.path.exists('./model_save/cls_model.pkl'):
            os.remove('./model_save/cls_model.pkl')
        torch.save(model, './model_save/cls_model.pkl')
    model = model.train()


set = textDataset(config.title_max_seqlen, config.max_seqlen, config.train_path)
train_size, validate_size = int(0.9 * len(set)), int(0.1 * len(set)+1)
train_set, val_set = torch.utils.data.random_split(set, [train_size, validate_size])
# model init
#model = att_cls()
# gpu or cpu
#if config.gpu:
#    model = model.cuda()
model = torch.load("./model_save/cls_model.pkl")

# cut into batche
loader_train = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=0, drop_last=True)
loader_val = DataLoader(val_set, batch_size=config.batch_size, drop_last=False)
# Adam
optimizer = optim.Adam(model.parameters(), config.lr)

train(model, loader_train, loader_val, optimizer)
print('the max_acc is:', config.max_acc)
