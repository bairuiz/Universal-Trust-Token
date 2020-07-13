from selenium import webdriver
from selenium.webdriver.edge.options import Options
from bs4 import BeautifulSoup
import time
import pandas as pd

def _login(browser, email, password):
    browser.get("http://facebook.com")
    browser.maximize_window()
    browser.find_element_by_name("email").send_keys(email)
    browser.find_element_by_name("pass").send_keys(password)
    browser.find_element_by_id('loginbutton').click()
    time.sleep(1)

def _extract_link(fb_url):
    time.sleep(1)
    browser.get(fb_url)
    bs_data = BeautifulSoup(browser.page_source, 'html.parser')
    try:
        link = bs_data.find(class_="_6ks").find('a').get('href')
        browser.get(link)
        news_link = browser.current_url
    except:
        news_link = ''
    return news_link

if __name__ == '__main__':
    
    # read post urls on facebook and labels
    data = pd.read_csv('facebook-fact-check.csv')
    data = data[data['Post Type'] == 'link'].reset_index(drop=True)
    
    # facebook account
    with open('facebook_credentials.txt') as file:
        EMAIL = file.readline().split('"')[1]
        PASSWORD = file.readline().split('"')[1]
    
    # log in facebook
    browser = webdriver.Edge(executable_path='./edgedriver_win64/msedgedriver.exe')
    _login(browser, EMAIL, PASSWORD)
    
    # extract news urls
    data['news_link'] = data['Post URL'].apply(_extract_link)
    
    # write to file
    data.to_csv('facebook_news.csv')
    
    # close the browser
    browser.quit()



