import torch
from lconfig import Config
from torch.utils.data import Dataset
import pandas as pd

"""
build pytorch dataset
"""

config = Config()


# textid turn word to id
def get_txtid(src_txt, src_w2id, max_seq):
    txtid = []
    for i in (src_txt.split()):
        if i in src_w2id:
            txtid.append(src_w2id[i])
        else:
            txtid.append(src_w2id['<pad>'])
    while (len(txtid) < max_seq):
        txtid.append(src_w2id['<pad>'])
    if (len(txtid) > max_seq):
        txtid = txtid[:max_seq]
    return txtid


# get train label
def get_trla(csv_path, w2id, title_max_seq, max_seq):
    title, text, label = [], [], []
    data = pd.read_csv(csv_path)
    for row in data.itertuples():
        src_title = getattr(row, 'title')
        temp1 = get_txtid(src_title, w2id, title_max_seq)
        # [1,4,2,5,7,0]
        if temp1==[0]*title_max_seq:
            continue
        src_text = getattr(row, 'text')
        temp2 = get_txtid(src_text, w2id, max_seq)
        if temp2==[0]*max_seq:
            continue
        title.append(temp1)
        text.append(temp2)
        label.append(int(getattr(row, 'label')))
    return title, text, label


class textDataset(Dataset):
    def __init__(self, title_max_seq, max_seq, path):
        title, text, label = get_trla(path, config.vob_w2id, title_max_seq, max_seq)
        #convert to pytorch tensor
        self.title = torch.tensor(title)
        self.text = torch.tensor(text)
        self.label = torch.tensor(label)

    def __getitem__(self, i):
        return (self.title[i], self.text[i], self.label[i])

    def __len__(self):
        return self.text.size(0)

# print(get_txtid1('i do not know',config.w2id,20))
