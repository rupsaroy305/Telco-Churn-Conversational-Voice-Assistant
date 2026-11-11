import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
from model_utils import FeatureEngineeringTransformer


NUMERIC_FEATURES = [
    "tenure", "MonthlyCharges", "TotalCharges", "AvgMonthlySpend"
]
CATEGORICAL_FEATURES = [
    "Contract", "InternetService", "TechSupport", "TenureGroup", "ChargeLevel"
]
MODEL_FEATURES = [
    "tenure", "MonthlyCharges", "TotalCharges",
    "Contract", "InternetService", "TechSupport"
]


def prepare_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(
        df["MonthlyCharges"] * df["tenure"]
    )
    return df[MODEL_FEATURES]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="Telco-Customer-Churn.csv")
    parser.add_argument("--output", default="models/churn_pipeline.joblib")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["Churn", "tenure", "MonthlyCharges"])
    df["TotalCharges"] = df["TotalCharges"].fillna(
        df["MonthlyCharges"] * df["tenure"]
    )

    X = prepare_raw_data(df)
    y = df["Churn"].map({"No": 0, "Yes": 1}).astype(int)

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    negatives = int((y_train == 0).sum())
    positives = int((y_train == 1).sum())

    pipeline = Pipeline([
        ("feature_engineering", FeatureEngineeringTransformer()),
        ("preprocessing", preprocessor),
        ("model", XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=negatives / max(positives, 1),
            random_state=42,
            n_jobs=-1
        ))
    ])
    pipeline.fit(X_train, y_train)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.40).astype(int)

    print("\nEvaluation at threshold 0.40")
    print(classification_report(y_test, predictions, zero_division=0))
    print("ROC-AUC:", round(roc_auc_score(y_test, probabilities), 4))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "threshold": 0.40}, args.output)
    print(f"\nSaved complete preprocessing + model pipeline to: {args.output}")


if __name__ == "__main__":
    main()
