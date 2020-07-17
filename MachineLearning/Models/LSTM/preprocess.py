import pandas as pd
import nltk
# nltk.download('punkt')
import numpy as np
import math


# Build a vocabulary
def build_vocab(csv_path, save_path):
    f1 = open(save_path, 'w+', encoding='utf-8')
    word_dic = {}
    data = pd.read_csv(csv_path)
    for row in data.itertuples():
        # get column of title
        title = getattr(row, 'title').strip().lower()
        # travel the list
        for word in nltk.tokenize.word_tokenize(title):
            if word in word_dic:
                word_dic[word] += 1
            else:
                word_dic[word] = 1
        # get text, same
        text = getattr(row, 'text').strip().lower()
        for word in nltk.tokenize.word_tokenize(text):
            if word in word_dic:
                word_dic[word] += 1
            else:
                word_dic[word] = 1
    for word, count in word_dic.items():
        if count >= 3:
            f1.write(word + '\n')
    f1.close()


build_vocab('./data/newData_w_title.csv', './data/vocab.txt')



def build_train_csv(csv_path):
    titles, texts, labels = [], [], []
    data = pd.read_csv(csv_path)
    for row in data.itertuples():
        title = getattr(row, 'title').strip().lower()
        text = getattr(row, 'text').strip().lower()
        title = ' '.join(nltk.tokenize.word_tokenize(title))
        text = ' '.join(nltk.tokenize.word_tokenize(text))
        if text == '':
            continue
        titles.append(title)
        texts.append(text)
        label = getattr(row, 'label')
        labels.append(label)
    dic = {'label': labels, 'title': titles, 'text': texts}
    data = pd.DataFrame(dic, columns=['label', 'title', 'text'])
    data = data.dropna(axis=0, how='any')
    data.to_csv('./data/train.csv', index=0)


build_train_csv('./data/newData_w_title.csv')

def get_mean_max(path):
    len_title, len_text = [], []
    data = pd.read_csv(path)
    for row in data.itertuples():
        title = getattr(row, 'title').strip()
        text = getattr(row, 'text').strip()
        len_title.append(len(nltk.tokenize.word_tokenize(title)))
        len_text.append(len(nltk.tokenize.word_tokenize(text)))

    a, b = np.array(len_title), np.array(len_text)
    print(a.mean(), a.max(-1))
    print(b.mean(), b.max(-1))

# get_mean_max('./data/train.csv')
