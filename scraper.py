import requests
from bs4 import BeautifulSoup

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'}
url = "https://finance.yahoo.com/quote/AAPL"

r = requests.get(url, headers=headers)
soup = BeautifulSoup(r.content, 'html.parser')

price = soup.find('fin-streamer', {'data-field': 'regularMarketPrice'}).text
change = soup.find('fin-streamer', {'data-field': 'regularMarketChange'}).text
percentage_change = soup.find('fin-streamer', {'data-field': 'regularMarketChangePercent'}).text

print(f"Price: {price}")
print(f"Change: {change}")
print(f"Percentage Change: {percentage_change}")

