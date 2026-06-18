import pandas as pd

r = pd.read_csv('fifa_ranking.csv', usecols=['rank', 'country_full', 'rank_date'])
r['rank_date'] = pd.to_datetime(r['rank_date'])
latest = r.sort_values('rank_date').groupby('country_full').tail(1)

teams = ['Germany', 'Brazil', 'Argentina', 'France', 'England', 'Belgium', 'Spain', 'Portugal']
for t in teams:
    row = latest[latest['country_full'] == t]
    if row.empty:
        print(f'{t}: NOT FOUND')
    else:
        rank = int(row.iloc[0]['rank'])
        date = row.iloc[0]['rank_date'].date()
        print(f'{t}: rank {rank}, last updated {date}')