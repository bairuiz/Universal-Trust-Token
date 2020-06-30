from newspaper import Article

def processUrl(url):
    if (url[:4] != "http"):
        url = "http://" + url
    article = Article(url)
    article.download()
    article.parse()
    return article
 
    