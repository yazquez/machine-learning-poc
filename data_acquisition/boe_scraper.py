import re
import requests
from bs4 import BeautifulSoup

url = "http://www.boe.es/boe_n/dias/2016/03/28/"
search_patterns = [
    '((.|\n)*)(Anuncio de notificación de)(.*)(en procedimientos tramitados por la Agencia Estatal de Administración Tributaria.*)$',
    '((.|\n)*)(Anuncio de notificación de)(.*)(en procedimiento Sancionador.*)$']

soup = BeautifulSoup(requests.get(url).text, 'html5lib')

all_notifications = soup('li', 'notif')

print("Número de notificaciones total: " , len(all_notifications))

links = []
num_notifications = 0
for notification in all_notifications:
    notification_text = notification.find('p').text
    for search_pattern in search_patterns:
        if (re.match(search_pattern, notification_text)):
            num_notifications = num_notifications + 1
            link = notification.find('a')
            link.notification_text = notification_text
            links.append(link)



for link in links:
    print(link.get('href'), " - ", link.get('title'), " - ", link.notification_text)
    # print(link.get('title'))

print("\nNúmero de notificaciones registradas: " , num_notifications)