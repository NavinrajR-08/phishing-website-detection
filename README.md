# Phishing Website Detection - Machine Learning (Python + scikit-learn)

A phishing URL detector trained on a real 10,000-URL dataset, using a
Random Forest classifier and 9 lexical (text-based) features extracted
straight from the URL - no internet lookups needed at prediction time.

## Project Structure
```
phishing_ml/
├── dataset/
│   └── urldata.csv          # 10,000 labeled URLs (5000 legit, 5000 phishing)
├── feature_extractor.py     # Turns a URL string into feature values
├── train_model.py           # Trains & evaluates the Random Forest model
├── predict.py                # Interactive CLI - classify URLs you type in
├── phishing_model.pkl        # Saved trained model (created after training)
└── requirements.txt
```

## Setup
```bash
pip install -r requirements.txt
```

## How to Run

**Step 1 - Train the model** (only needs to be done once; it saves
`phishing_model.pkl` to disk):
```bash
python train_model.py
```
This prints the accuracy, a classification report, a confusion matrix,
and feature importances - all useful to include in your project report.

**Step 2 - Classify URLs:**
```bash
python predict.py
```
Type any URL when prompted and it will tell you Legitimate or Phishing,
along with the model's confidence percentage.

## Dataset Source
Built from PhishTank (phishing URLs) and the University of New
Brunswick's URL dataset (legitimate URLs), with features referenced
from the UCI "Phishing Websites" dataset. Original compiled dataset:
https://github.com/shreyagopal/Phishing-Website-Detection-by-Machine-Learning-Techniques

## Features Used
All 9 features are computed purely from the domain name text - no
website visit, WHOIS lookup, or DNS query required:

| Feature | What it checks |
|---|---|
| Have_IP | Is the domain a raw IP address? |
| Have_At | Does the URL contain '@'? |
| Domain_Length | Is the domain unusually long (>= 20 chars)? |
| Count_Dots | How many '.' (subdomains) in the domain? |
| Count_Hyphens | How many '-' (often used to fake brand names)? |
| Has_Digit | Does the domain contain digits? |
| Https_Token_In_Domain | Does 'https' appear inside the domain itself (a trick)? |
| TinyURL | Is a known URL-shortening service used? |
| Suspicious_Keyword | Does the domain contain words like "login", "secure", "verify"? |

**Why only lexical features?** The original dataset also includes
domain-age, DNS record, and web-traffic based features, but those
require live WHOIS/DNS lookups and an internet connection at prediction
time, which makes a project unreliable to demo (and slow). Restricting
to lexical, string-only features keeps the tool instant and 100%
offline, at the cost of some accuracy.

## Model Performance
Random Forest, 80/20 train-test split:
- **Accuracy: ~75%**
- Domain length, number of dots, and number of hyphens are the three
  most important features (see `feature_importances_` output).

## Known Limitation (worth mentioning in your report/viva)
Because the model only looks at the domain name text, it can
occasionally misjudge legitimate sites with slightly unusual domain
structure (e.g. multiple subdomains) as suspicious, and it cannot catch
phishing pages hosted on otherwise "clean-looking" domains (e.g. a
phishing page hosted on a compromised legitimate site). A production
system would combine this with domain-age/DNS/web-traffic checks and
page content analysis for higher accuracy.

## Possible Extensions
- Add a Tkinter or Flask front-end so users can type a URL into a text box
- Compare multiple models (Decision Tree, SVM, XGBoost) like the original
  dataset's authors did, and report which performs best
- Add domain-age/WHOIS features (requires internet access) for a
  "hybrid" version
