import re

import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

df = pd.read_html('https://www.imdb.com/chart/top')[0]
df.drop(['Unnamed: 0', 'Unnamed: 4', 'Your Rating'], axis=1, inplace=True)
df['matched'] = df.apply(lambda r: list(re.match(r'(\d+)\. ([\w\'\s:,-·.]+) \((\d{4})\)', r['Rank & Title']).groups()), axis=1)
df[['rank', 'title', 'year']] = pd.DataFrame(df['matched'].tolist(), index=df.index)
df[['rank', 'year', 'IMDb Rating']] = df[['rank', 'year', 'IMDb Rating']].apply(pd.to_numeric)
df.drop(['Rank & Title', 'matched'], axis=1, inplace=True)
df = df[['rank', 'title', 'year', 'IMDb Rating']]
print(df.head(100))


def rank_of_nth_oldest(data, n):
    sorted_by_year = data.copy().sort_values(by='year').reset_index(drop=True)
    print(sorted_by_year.head(100))
    return sorted_by_year.loc[n - 1, 'rank']


def count_movies_by_rating(data, condition):
    return len(list(filter(condition, data['IMDb Rating'])))


ans1 = rank_of_nth_oldest(df, 75)
ans2 = rank_of_nth_oldest(df, 110)
ans3 = sum(df['year'])
ans4 = count_movies_by_rating(df, lambda r: r >= 8.5)
ans5 = count_movies_by_rating(df, lambda r: r <= 8.2)
