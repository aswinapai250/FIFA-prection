import joblib
import pandas as pd
import numpy as np

model = joblib.load('model.pkl')
feature_columns = joblib.load('feature_columns.pkl')
data = pd.read_csv('wc_features_ranked.csv')

probs = model.predict_proba(data[feature_columns])[:, 1]
print(f"Min:    {probs.min():.4f}")
print(f"Max:    {probs.max():.4f}")
print(f"Mean:   {probs.mean():.4f}")
print(f"Median: {np.median(probs):.4f}")
print(f"% above 0.7: {(probs > 0.7).mean()*100:.1f}%")
print(f"% below 0.3: {(probs < 0.3).mean()*100:.1f}%")