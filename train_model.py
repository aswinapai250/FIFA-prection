import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier

INPUT_FILE = "all_features.csv"
MODEL_FILE = "model.pkl"
FEATURE_COLUMNS_FILE = "feature_columns.pkl"

FEATURE_COLUMNS = [
    "home_win_rate",
    "away_win_rate",
    "home_goals_avg",
    "away_goals_avg",
    "home_conceded_avg",
    "away_conceded_avg",
    "h2h_home_wins",
    "home_rank",
    "away_rank",
    "rank_diff",
    "home_win_rate_vs_top",
    "away_win_rate_vs_top",
]
TARGET_COLUMN = "label"
TEST_YEAR = 2022


def main():
    data = pd.read_csv(INPUT_FILE)
    data["year"] = pd.to_datetime(data["date"]).dt.year

    train_data = data[data["year"] < TEST_YEAR]
    test_data = data[data["year"] == TEST_YEAR]

    x_train = train_data[FEATURE_COLUMNS]
    y_train = train_data[TARGET_COLUMN]
    x_test = test_data[FEATURE_COLUMNS]
    y_test = test_data[TARGET_COLUMN]

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )
    calib_size = int(len(x_train) * 0.2)
    x_calib = x_train.iloc[-calib_size:]
    y_calib = y_train.iloc[-calib_size:]
    x_fit = x_train.iloc[:-calib_size]
    y_fit = y_train.iloc[:-calib_size]

    model.fit(x_fit, y_fit)

    calibrated_model = CalibratedClassifierCV(
        FrozenEstimator(model), method="sigmoid"
    )
    calibrated_model.fit(x_calib, y_calib)

    y_pred = calibrated_model.predict(x_test)
    test_proba = calibrated_model.predict_proba(x_test)[:, 1]

    print(f"Train rows: {len(train_data)} (before {TEST_YEAR})")
    print(f"Test rows: {len(test_data)} ({TEST_YEAR} only)")
    print()
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1 score:  {f1_score(y_test, y_pred):.4f}")
    print()
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print()
    print("Test set home-win probability distribution:")
    print(f"  Min:    {test_proba.min():.4f}")
    print(f"  Max:    {test_proba.max():.4f}")
    print(f"  Mean:   {test_proba.mean():.4f}")
    print(f"  Median: {np.median(test_proba):.4f}")
    print()
    print("Feature importances:")
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
    for feature, importance in importances.sort_values(ascending=False).items():
        print(f"  {feature}: {importance:.4f}")

    joblib.dump(calibrated_model, MODEL_FILE)
    joblib.dump(FEATURE_COLUMNS, FEATURE_COLUMNS_FILE)
    print()
    print(f"Saved model to {MODEL_FILE}")
    print(f"Saved feature columns to {FEATURE_COLUMNS_FILE}")


if __name__ == "__main__":
    main()
