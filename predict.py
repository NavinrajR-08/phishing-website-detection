"""
predict.py
----------
Loads the trained model and classifies URLs typed in by the user as
'Phishing' or 'Legitimate', along with the model's confidence.

Run:
    python predict.py
"""

import joblib
import pandas as pd
from feature_extractor import extract_features, FEATURE_NAMES

MODEL_PATH = "phishing_model.pkl"


def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"Error: '{MODEL_PATH}' not found.")
        print("Run 'python train_model.py' first to train and save the model.")
        exit(1)


def predict_url(model, url: str):
    features = extract_features(url)
    # Wrap as a DataFrame with matching column names (sklearn expects this
    # to avoid a 'feature names mismatch' warning)
    X = pd.DataFrame([features], columns=FEATURE_NAMES)

    prediction = model.predict(X)[0]          # 0 = Legitimate, 1 = Phishing
    probabilities = model.predict_proba(X)[0]  # [prob_legit, prob_phish]

    return prediction, probabilities, dict(zip(FEATURE_NAMES, features))


def main():
    model = load_model()

    print("=================================================")
    print("     PHISHING URL DETECTION (ML - Random Forest)")
    print("=================================================")

    while True:
        url = input("\nEnter a URL to check (or 'exit' to quit): ").strip()

        if url.lower() == "exit":
            print("Goodbye!")
            break
        if not url:
            print("Please enter a valid URL.")
            continue

        prediction, probs, features = predict_url(model, url)

        print("\n---------------------------------------------")
        print(f"URL: {url}")
        print("Extracted features:", features)
        print(f"Confidence -> Legitimate: {probs[0]*100:.1f}% | Phishing: {probs[1]*100:.1f}%")

        if prediction == 1:
            print("VERDICT: Likely PHISHING website!")
        else:
            print("VERDICT: Likely LEGITIMATE website.")
        print("---------------------------------------------")


if __name__ == "__main__":
    main()
