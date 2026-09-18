import pandas as pd
from bs4 import BeautifulSoup
import requests

mystery_response=requests.get("https://books.toscrape.com/catalogue/category/books/mystery_3/index.html").text

soup=BeautifulSoup(mystery_response,'lxml')

mystery =[]
# print(len(soup.select("ol li")))
for i in soup.select("ol li"):

    mystery.append(i.select("img")[0].get('alt'))
print(mystery)

df = pd.DataFrame(mystery, columns=["Title"])

df.to_csv("mystery.csv", index=False)
