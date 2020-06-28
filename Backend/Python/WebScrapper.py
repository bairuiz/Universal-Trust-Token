import requests
from bs4 import BeautifulSoup

def parse(url):
    if (url[:4] != "http"):
        url = "http://" + url
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    print(soup.prettify()[:200])
    