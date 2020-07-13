# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 00:10:03 2020

@author: lixum
"""
#
# TBD#FF5722
# Error codes:
# O - OK
# C - Connection timeout
# I - Invalid URL
# N - Invalid News URL
# U - Unkown Error
#
import WebScrapper as ws
import sys
try:
    url = sys.argv[1]
    news = ws.processUrl(url)
    print("O")
    print("%3dAuthor:%s\nPublish Date:%s\nTitle:%s\n"
          %(80, news.authors, news.publish_date, news.title))
except:
    print("U")