import pandas as pd
from newspaper import Article

def newsScrapping(url):
    article = Article(url)
    article.download()
    article.parse()
    return article

if __name__ == '__main__':
    
    # read news urls and labels
    data = pd.read_csv('facebook_news.csv', index_col=0)
    
    # remove posts without news urls
    data = data[data['news_link'].notnull()]
    data = data.reset_index(drop=True)
    
    # add fake/true labels
    data.loc[data['Rating'] == 'mostly true', 'label'] = 1
    data.loc[data['Rating'] != 'mostly true', 'label'] = 0
    
    # make label into categorical data
    data['label'] = data['label'].astype('category')
        
    # get news title and text            
 #   data = data.head(10)
    for i, url in zip(range(len(data)), data['news_link']):
        try:
            print(i)
            article = newsScrapping(url)
            data.loc[i, 'title'] = article.title
            data.loc[i, 'text'] = article.text
            data.loc[i, 'date'] = data.loc[i, 'Date Published']
        except:
            continue
    
    # select columns and remove Nulls
    data = data[['title', 'text', 'date', 'label']]
    data = data.dropna()
    data = data.reset_index(drop=True)
    
    data.to_csv('fb_news_text.csv')


    
    