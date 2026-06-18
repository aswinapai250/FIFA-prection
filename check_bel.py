import joblib
import pandas as pd

model = joblib.load('model.pkl')
feature_cols = joblib.load('feature_columns.pkl')
features = pd.read_csv('all_features_ranked.csv')

# Check Belgium's average feature values
bel = features[features['home_team'] == 'Belgium'][feature_cols].mean()
print("Belgium avg features:")
print(bel)