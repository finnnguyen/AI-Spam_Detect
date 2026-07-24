# Spam Tool Kit

An AI-powered web application that detects spam in emails and SMS messages, with additional tools for text analysis and plagiarism detection — deployed as a live Flask web app.

**Tech:** Python · scikit-learn · Flask · CountVectorizer · Multinomial Naive Bayes · pandas · NumPy · HTML · CSS · JavaScript

---

## Features

| Module | What it does |
|---|---|
| **Email Spam** | Upload a `.txt` or `.eml` file — returns spam/ham label, confidence score, and risk level |
| **SMS Spam** | Paste any SMS message — same output as email module |
| **Text Analysis** | Word count, character count, sentence count, unique words, avg word length, most common words |
| **Plagiarism Detection** | Compare two texts using Jaccard similarity — returns similarity % and plagiarism level |

---

## Model Performance

Trained on Kaggle's `email_spam.csv` and `sms_spam.csv` datasets with an 80/20 train/test split:

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Email Spam (Naive Bayes) | 97.20% | — | — | — |
| SMS Spam (Naive Bayes) | **99.28%** | **100%** | 94.63% | 97.24% |

- Confusion matrix (SMS): 966 true positives, 0 false negatives, 1 false positive
- Inference speed: under 1–2ms per message
- Training time: under 1 second

---

## Pipeline

```
Raw text → Data cleaning → Label encoding (spam=1, ham=0)
→ CountVectorizer (bag-of-words) → Multinomial Naive Bayes
→ Spam/Ham label + Probability score + Confidence level + Risk category
```

---

## Project Structure

```
AI-Spam_Detect/
├── src/
│   └── App.py              # Flask server, model training, prediction routes
├── templates/
│   └── indextesting.html   # Web UI (HTML/CSS/JS)
├── data/
│   └── csv/
│       ├── email_spam.csv  # Kaggle email spam dataset
│       └── sms_spam.csv    # Kaggle SMS spam dataset
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/finnnguyen/AI-Spam_Detect
cd AI-Spam_Detect
pip install -r requirements.txt
python src/App.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## Credits

Built by **Finn Nguyen** (all technical implementation — model, backend, and UI) as part of a group project.  
CPSC 483 · Cal State Fullerton

---

[→ Featured on my portfolio](https://finn-portfolio-phi.vercel.app)
