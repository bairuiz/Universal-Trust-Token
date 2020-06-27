import sys
import requests
from bs4 import BeautifulSoup

url = sys.argv[1]
if (url[:4] != "http"):
    url = "http://"+url
page = requests.get(url)
soup = BeautifulSoup(page.content, 'html.parser')
print(soup.prettify()[:200])