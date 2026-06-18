import joblib
import pandas as pd

r = pd.read_csv('fifa_ranking.csv', usecols=['rank', 'country_full', 'rank_date'])
r['rank_date'] = pd.to_datetime(r['rank_date'])
latest = r.sort_values('rank_date').groupby('country_full').tail(1)
rankings = dict(zip(latest['country_full'], latest['rank']))

teams = ['Germany', 'Brazil', 'Argentina', 'France', 'England', 'Belgium', 'Spain', 'Portugal']
for t in teams:
    print(f'{t}: {rankings.get(t, "NOT FOUND")}')