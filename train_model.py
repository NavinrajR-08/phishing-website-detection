"""
train_model.py
--------------
Trains a Random Forest classifier on the phishing URL dataset and
saves the trained model to disk for use by predict.py.

Dataset source: shreyagopal/Phishing-Website-Detection-by-Machine-Learning-Techniques
(originally built from PhishTank + University of New Brunswick URL datasets,
using features referenced from the UCI Phishing Websites dataset)

We only use the 8 LEXICAL features (computable from the URL string alone),
so the trained model can classify brand-new URLs instantly and offline,
without needing to visit the website, query DNS, or check WHOIS records.

Run:
    python train_model.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

from feature_extractor import FEATURE_NAMES, extract_features

DATA_PATH = "dataset/urldata.csv"
MODEL_PATH = "phishing_model.pkl"


def main():
    print("Loading dataset...")
    raw = pd.read_csv(DATA_PATH)

    # NOTE: the CSV's own pre-computed feature columns (URL_Length, URL_Depth,
    # etc.) were calculated from the ORIGINAL full URLs, which are not stored
    # in this file - only the bare "Domain" is. If we trained on those stored
    # columns but extracted features live from just a domain the user types,
    # the two would be inconsistent and predictions would be unreliable.
    #
    # So instead we recompute every feature ourselves directly from the
    # "Domain" column, using the exact same function the live predictor
    # uses. This guarantees training and prediction always match.
    print("Recomputing features from domain strings (for train/predict consistency)...")
    feature_rows = raw["Domain"].apply(extract_features)
    df = pd.DataFrame(feature_rows.tolist(), columns=FEATURE_NAMES)
    df["Label"] = raw["Label"]

    print(f"Dataset shape: {df.shape}")
    print(f"Class balance:\n{df['Label'].value_counts()}\n")

    X = df[FEATURE_NAMES]
    y = df["Label"]

    # 80-20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n================ EVALUATION ================")
    print(f"Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"]))
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, y_pred))

    # Feature importance - useful to include in your project report
    print("\nFeature Importances:")
    importances = sorted(
        zip(FEATURE_NAMES, model.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    for name, imp in importances:
        print(f"  {name:15s}: {imp:.4f}")

    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to '{MODEL_PATH}'")


if __name__ == "__main__":
    main()
