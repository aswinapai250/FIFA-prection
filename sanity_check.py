import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv('all_features_ranked.csv')
data['year'] = pd.to_datetime(data['date']).dt.year
train = data[data['year'] < 2022]
test = data[data['year'] == 2022]

feature_cols = joblib.load('feature_columns.pkl')
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(train[feature_cols], train['label'])
pred = dummy.predict(test[feature_cols])
print(f'Dummy classifier accuracy: {accuracy_score(test["label"], pred):.4f}')
print('Your model must beat this to be useful')