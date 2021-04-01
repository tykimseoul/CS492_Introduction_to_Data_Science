import re

import matplotlib.pyplot as plt
import pandas as pd
import requests
from bs4 import BeautifulSoup

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

html = requests.get('https://www.imdb.com/chart/top')
doc = BeautifulSoup(html.text, "html.parser")
poster_column = doc.select('tbody.lister-list > tr > td.posterColumn')
data = list(map(lambda p: dict(map(lambda r: (r['name'], r['data-value']), p.select('span'))), poster_column))
df = pd.DataFrame.from_records(data)
title_column = doc.select('tbody.lister-list > tr > td.titleColumn')
titles = list(map(lambda t: re.sub(r'\s+', ' ', t.text.strip()), title_column))
df['Rank & Title'] = titles
df.drop(['nv', 'ur'], axis=1, inplace=True)
df['matched'] = df.apply(lambda r: list(re.match(r'\d+\. ([\w\'\s:,-·.]+) \((\d{4})\)', r['Rank & Title']).groups()), axis=1)
df[['title', 'year']] = pd.DataFrame(df['matched'].tolist(), index=df.index)
df[['year', 'rk', 'ir', 'us']] = df[['year', 'rk', 'ir', 'us']].apply(pd.to_numeric)
df.drop(['Rank & Title', 'matched'], axis=1, inplace=True)
df = df[['rk', 'title', 'year', 'ir', 'us']]
df.plot.scatter(x='us', y='year', c='DarkBlue')
plt.show()
df.rename(columns={'rk': 'rank', 'ir': 'rating', 'us': 'release date'}, inplace=True)
print(df.head(10))


def rank_of_nth_oldest(data, n):
    sorted_by_year = data.copy().sort_values(by='release date').reset_index(drop=True)
    print(sorted_by_year.head(100))
    return sorted_by_year.loc[n - 1, 'rank']


def count_movies_by_rating(data, condition):
    return len(list(filter(condition, data['rating'])))


ans1 = rank_of_nth_oldest(df, 75)
ans2 = rank_of_nth_oldest(df, 110)
ans3 = sum(df['year'])
ans4 = count_movies_by_rating(df, lambda r: r >= 8.5)
ans5 = count_movies_by_rating(df, lambda r: r <= 8.2)
