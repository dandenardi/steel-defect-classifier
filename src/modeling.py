import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from utils.data_loader import load_data

DATA_PATH = Path("../data/processed/cleaned_data.csv")

def split_features_targets(df):
    target_cols = [col for col in df.columns if col.startswith("falha_")]
    X = df.drop(columns=target_cols + ["id"])
    y = df[target_cols]
    print(y.head())

    return X, y

def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("\n Confusion Matrix (multilabel):")
    print(multilabel_confusion_matrix(y_test, y_pred))

    joblib.dump(model, "models/random_forest_model.pkl")

    return model

def load_trained_model():
    return joblib.load("models/random_forest_model.pkl")

if __name__ == "__main__":
    df = load_data(DATA_PATH)
    print(df.head())

    X,y = split_features_targets(df)
    trained_model = train_and_evaluate(X, y)
    model = load_trained_model()