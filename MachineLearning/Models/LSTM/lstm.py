import pandas as pd
from textblob import Word
from newspaper import Article
from model.classify import att_cls
from dataprocess import get_txtid
from lconfig import Config
import torch

config = Config()

def test(model, title, text):
    title_id = get_txtid(title, config.vob_w2id, config.title_max_seqlen)
    text_id = get_txtid(text, config.vob_w2id, config.max_seqlen)
    title_id, text_id = torch.tensor([title_id]), torch.tensor([text_id])
    if config.gpu:
        title_id, text_id = title_id.cuda(), text_id.cuda()
    confid = model(title_id, title_id.eq(0), text_id, text_id.eq(0))
    return confid


def clean_dataset(X):
    # remove digits
    # remove words less than 3 characters
    # remove punctuation

    X['clean_title'] = X['title'].str.replace('\d+', ' ')  # for digits
    X['clean_title'] = X['clean_title'].str.replace(r'(\b\w{1,2}\b)', ' ')  # for words less than 3 characters
    X['clean_title'] = X['clean_title'].str.replace('[^\w\s]', ' ')  # for punctuation

    X['clean_text'] = X['text'].str.replace('\d+', ' ')  # for digits
    X['clean_text'] = X['clean_text'].str.replace(r'(\b\w{1,2}\b)', ' ')  # for words less than 3 characters
    X['clean_text'] = X['clean_text'].str.replace('[^\w\s]', ' ')  # for punctuation
    # lemmatization
    X['clean_title'] = X['clean_title'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    X['clean_text'] = X['clean_text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    return X

def predict_proba(df, model):
    return test(model, df.loc[0]['clean_title'], df.loc[0]['clean_text']).tolist()[0][0]

if __name__ == '__main__':
    # Simulating parameters from PythonServer
    url = 'https://www.foxnews.com/politics/supreme-court-strikes-down-state-ban-on-taxpayer-funding-for-religious-schools'
    article = Article(url)
    article.download()
    article.parse()
    articles = {'title': [article.title], 'text': [article.text]}
    df = pd.DataFrame(articles)
    df = clean_dataset(df)
    model = torch.load(config.cls_model_path)

    # pred:[probability of true, probability of false]
    pred = predict_proba(df, model)
    print("Trust rating: %.0f" % (pred * 100))
