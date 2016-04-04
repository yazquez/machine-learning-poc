# 20 minutos example - First part
# Look at the terms and the robots.txt file
from bs4 import BeautifulSoup
import requests
user_agent = 'Mozilla/5.0 (X11; Linux x86_64; rv:38.0) Gecko/20100101 Firefox/38.0'
headers = {
    'User-Agent': user_agent
}

url = "http://jobs.monster.co.uk/search/?q=java&where=Dublin"
soup = BeautifulSoup(requests.get(url).text, 'html5lib')


search_results = soup('h2', 'page-title hidden-xs')

for link in search_results:
    print(link)
    print(link.text.strip().split()[0])
