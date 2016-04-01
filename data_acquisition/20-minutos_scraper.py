# 20 minutos example - First part
# Look at the terms and the robots.txt file
from bs4 import BeautifulSoup
import requests
user_agent = 'Mozilla/5.0 (X11; Linux x86_64; rv:38.0) Gecko/20100101 Firefox/38.0'
headers = {
    'User-Agent': user_agent
}
url = "http://www.20minutos.es"
soup = BeautifulSoup(requests.get(url).text, 'html5lib')
links = []

all_news_lines = soup('div', 'sep-top')

for line in all_news_lines:
    link = line.find('a')
    links.append(link)


news = []
for link in links:
    new = link.get('title')
    news.append(new)
    
print(news)