from newspaper import Article

#url = 'https://www.reuters.com/article/us-usa-afghanistan-russia/white-house-to-brief-democrats-on-alleged-russian-payments-to-kill-u-s-troops-idUSKBN241242'
#url = 'https://www.cnn.com/travel/article/eu-borders-open-but-not-to-americans-intl/index.html'
url = 'https://www.foxnews.com/politics/supreme-court-strikes-down-state-ban-on-taxpayer-funding-for-religious-schools'
#url = 'https://www.nbcnews.com/news/world/european-union-bars-travelers-u-s-citing-coronavirus-concerns-n1232333'
article = Article(url)

article.download()
print('article.html:', article.html[0:5000], sep='\n', end='\n\n')


article.parse()

print('article.authors:', article.authors, sep='\n', end='\n\n')
print('article.publish_date:', article.publish_date, sep='\n', end='\n\n')
print('article.title:', article.title, sep='\n', end='\n\n')
print('article.text:', article.text, sep='\n', end='\n\n')

article.nlp()
print('article.keywords:', article.keywords, sep='\n', end='\n\n')
print('article.summary:', article.summary, sep='\n', end='\n\n')

