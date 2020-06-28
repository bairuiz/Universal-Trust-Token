# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 00:10:03 2020

@author: lixum
"""
import WebScrapper
import sys
url = sys.argv[1]
print("%s%3dREPLY from windows server. Testing Analysis." %("F", 80))
WebScrapper.parse(url)