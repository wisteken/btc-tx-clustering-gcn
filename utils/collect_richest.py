import requests
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup


addresses = []
pages = [''] + [f'-{i}' for i in range(2, 11)]
for page in tqdm(pages):
    url = f'https://bitinfocharts.com/top-100-richest-bitcoin-addresses{page}.html'
    html = requests.get(url)
    soup = BeautifulSoup(html.content, 'html.parser')
    table_top = soup.find('table', class_="table table-striped abtb")
    for tr in table_top.find_all('tr'):
        link = tr.find('a')
        if (link):
            addresses.append(link.get('href').split('/')[-1])
    table_bottom = soup.find('table', class_="table table-striped bb")
    for tr in table_bottom.find_all('tr'):
        link = tr.find('a')
        if (link):
            addresses.append(link.get('href').split('/')[-1])

df = pd.DataFrame(addresses, columns=['address'], index=range(1, len(addresses) + 1))
df.to_csv('../datasets/addresses.csv', index=True)
