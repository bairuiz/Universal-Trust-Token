from newspaper import Article
import TrustCal as tc
import pandas as pd
from Log import log

log('Loading module data...')
cv_vector_title, cv_vector_text, tfidf_vector_title, tfidf_vector_text = tc.loadVectors()
svm,rf,lr,mlp,lstm = tc.loadModels()
log('Loading complete.')

def download(url):
    if (url[:4] != "http"):
        url = "http://" + url
    try:
        news = Article(url)
        news.download()
        news.parse()
        return news
    except:
        return None

def process(news):
    try:
        #parse information into a dataframe
        article = {'title': [news.title],'text': [news.text]}
        df = pd.DataFrame(article)
        #calculate percentage of real
        title = news.title
        percentage = tc.calculate(df,cv_vector_title, cv_vector_text, tfidf_vector_title, tfidf_vector_text, svm, rf, lr, mlp, lstm)
        return title, percentage
    except:
        return '', -1
