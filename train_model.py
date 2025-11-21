"""
Train a simple classifier on the collected landmark CSV and save the model.

Usage:
  python train_model.py --input data.csv --output model.pkl
"""
import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score


def main(input_csv: str, output_model: str):
    if not os.path.exists(input_csv):
        print(f"Input CSV not found: {input_csv}")
        sys.exit(1)

    df = pd.read_csv(input_csv)
    if df.empty:
        print("No data found in CSV")
        sys.exit(1)

    labels = df['label'].values
    X = df.drop(columns=['label']).values

    le = LabelEncoder()
    y = le.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=42))
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test accuracy: {acc:.3f}")
    print(classification_report(y_test, preds, target_names=le.classes_))

    # save model and label encoder
    out = {
        'model': clf,
        'label_encoder': le,
    }
    joblib.dump(out, output_model)
    print(f"Saved model to {output_model}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data.csv', help='Input CSV file')
    parser.add_argument('--output', default='model.pkl', help='Output model file (joblib)')
    args = parser.parse_args()
    main(args.input, args.output)
