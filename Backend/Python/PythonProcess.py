# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 00:10:03 2020

@author: lixum
"""
import WebScrapper as ws
import sys
url = sys.argv[1]
news = ws.processUrl(url)
print("%s%3dAuthor:%s\nPublish Date:%s\nTitle:%s\n"
      %("T", 80, news.authors, news.publish_date, news.title))