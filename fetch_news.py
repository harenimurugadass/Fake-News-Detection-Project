import requests
import pandas as pd

API_KEY = 'bd3977bc4a2943eea17debffa73615a8'
url = ('https://newsapi.org/v2/top-headlines?'
       'sources=bbc-news,cnn,reuters&'
       'pageSize=50&'
       f'apiKey={API_KEY}')

response = requests.get(url)
data = response.json()

articles = data['articles']
titles = [article['title'] for article in articles]
contents = [article['content'] for article in articles]

df_real = pd.DataFrame({'title': titles, 'text': contents, 'label': 1})
df_real.to_csv('additional_real_news.csv', index=False)

print("additional_real_news.csv created successfully.")
