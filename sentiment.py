"""
sentiment.py
------------
All the "machine learning" code of VoiceOfcampus lives here:
  1. Loading the pretrained CardiffNLP RoBERTa model
  2. Predicting sentiment (Positive / Neutral / Negative) with confidence
  3. Evaluating predictions against labelled data (scikit-learn)
  4. Simple keyword extraction for common issues

NOTE: The model is PRETRAINED (trained by CardiffNLP on tweets).
We do NOT fine-tune it on our college data.
"""

import re
from collections import Counter

import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
LABELS = ["Positive", "Neutral", "Negative"]  # fixed order used everywhere
MAX_LENGTH = 128  # longer feedback is truncated to 128 tokens


# ---------------------------------------------------------------------------
# 1. Loading the model
# ---------------------------------------------------------------------------
def load_model():
    """Download (first time only) and load the tokenizer + model.
    Returns (tokenizer, model). app.py calls this only once."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()  # inference mode (no training)

    # Safety check: make sure the model's output IDs map to our 3 labels
    names = {name.capitalize() for name in model.config.id2label.values()}
    if names != set(LABELS):
        raise ValueError(f"Unexpected model labels: {model.config.id2label}")
    return tokenizer, model


# ---------------------------------------------------------------------------
# 2. Prediction
# ---------------------------------------------------------------------------
def clean_text(text):
    """Return cleaned text, or None if the input is empty / invalid."""
    if not isinstance(text, str):  # handles None, NaN, numbers
        return None
    text = " ".join(text.split())  # remove extra spaces and new lines
    if len(text) < 3 or not any(ch.isalpha() for ch in text):
        return None
    return text


def predict_batch(texts, tokenizer, model, batch_size=16):
    """Run the RoBERTa model on a list of (already cleaned) texts.
    Returns a list of dicts, one per text:
      {"label": "Positive", "confidence": 97.31,
       "scores": {"Positive": 97.31, "Neutral": 2.1, "Negative": 0.59}}
    """
    # model.config.id2label looks like {0: 'negative', 1: 'neutral', 2: 'positive'}
    id_to_label = {i: name.capitalize() for i, name in model.config.id2label.items()}
    results = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        # Convert text to token IDs
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        # Real model inference (no gradient calculation needed)
        with torch.no_grad():
            logits = model(**inputs).logits
        # Softmax turns raw scores into probabilities that add up to 1
        probs = torch.softmax(logits, dim=1)

        for row in probs:
            scores = {id_to_label[i]: round(float(p) * 100, 2) for i, p in enumerate(row)}
            best_id = int(torch.argmax(row))
            results.append({
                "label": id_to_label[best_id],
                "confidence": round(float(row[best_id]) * 100, 2),
                "scores": scores,
            })
    return results


# ---------------------------------------------------------------------------
# 3. Evaluation (uses scikit-learn only for the metrics)
# ---------------------------------------------------------------------------
def evaluate_predictions(actual, predicted):
    """Compare actual labels (from CSV) with model predictions.
    Returns a dictionary with overall metrics, a per-class table and the
    confusion matrix (rows = actual, columns = predicted)."""
    accuracy = accuracy_score(actual, predicted)

    # Macro = simple average over the 3 classes
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        actual, predicted, labels=LABELS, average="macro", zero_division=0
    )
    # Per-class numbers
    p, r, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=LABELS, average=None, zero_division=0
    )
    per_class = pd.DataFrame({
        "Class": LABELS,
        "Precision": p.round(3),
        "Recall": r.round(3),
        "F1-score": f1.round(3),
        "Support": support,
    })
    cm = confusion_matrix(actual, predicted, labels=LABELS)
    cm_df = pd.DataFrame(cm, index=LABELS, columns=LABELS)

    return {
        "accuracy": accuracy,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": cm_df,
    }


# ---------------------------------------------------------------------------
# 4. Common issues (simple word / word-pair counting, NOT topic modelling)
# ---------------------------------------------------------------------------
# Common English words that carry no meaning for issue detection
STOPWORDS = set("""
a about after again all also am an and any are aren't as at be because been before being both but by can can't
could couldn't did didn't do does doesn't doing don't down during each even every few for from further get got
had has have having he her here hers him his how i if in into is isn't it its just like lot me more most much my
need needs never no nor not now of off on once one only or other our ours out over own same she should so some
still such than that the their theirs them then there these they this those through to too under until up us very
was wasn't we were what when where which while who whom why will with won't would you your really always
please many students student things thing something anything nothing hardly rarely often sometimes already
usually without despite instead whenever whether since though although however
another someone finally keep make makes hope better
""".split())


def extract_common_issues(texts, top_n=10):
    """Count the most frequent meaningful words and two-word phrases.
    Returns (top_words, top_phrases) as lists of (text, count)."""
    word_counter = Counter()
    phrase_counter = Counter()

    for text in texts:
        text = str(text).lower().replace("wi-fi", "wifi")
        tokens = re.findall(r"[a-z']+", text)
        # Keep meaningful words, use None as a placeholder for removed words
        meaningful = [t if (t not in STOPWORDS and len(t) > 2) else None for t in tokens]

        for word in meaningful:
            if word:
                word_counter[word] += 1
        # Two meaningful words next to each other form a phrase, e.g. "lab equipment"
        for first, second in zip(meaningful, meaningful[1:]):
            if first and second:
                phrase_counter[first + " " + second] += 1

    return word_counter.most_common(top_n), phrase_counter.most_common(top_n)


# ---------------------------------------------------------------------------
# 5. Issue category (TOPIC of the feedback, found by simple keyword matching)
#    NOTE: this only says WHAT the feedback is about.
#    The SENTIMENT always comes from the RoBERTa model, never from keywords.
# ---------------------------------------------------------------------------
# The order matters: if two categories have the same score, the first one wins.
CATEGORY_KEYWORDS = {
    "Wi-Fi / Internet": ["wifi", "internet", "network", "bandwidth", "router", "connectivity", "online"],
    "Mess / Food": ["canteen", "mess", "food", "meal", "thali", "tea", "snack", "menu", "sandwich",
                    "juice", "lunch", "breakfast", "dinner"],
    "Cleanliness": ["washroom", "toilet", "restroom", "bathroom", "sanitary", "soap", "hygiene", "hygienic",
                    "dirty", "filthy", "smell", "garbage", "litter", "cleanliness", "clean", "unclean",
                    "cleaning", "dustbin"],
    "Library": ["library", "librarian", "book", "journal", "reading", "catalog"],
    "Laboratory": ["lab", "laboratory", "practical", "experiment", "equipment", "apparatus", "instrument",
                   "component"],
    "Hostel": ["hostel", "warden", "geyser", "laundry", "dorm"],
    "Classroom": ["classroom", "projector", "bench", "whiteboard", "board", "fan", "seminar", "desk"],
    "Sports": ["sport", "gym", "playground", "football", "cricket", "badminton", "basketball", "coach",
               "swimming", "tournament"],
    "Transport": ["bus", "transport", "driver", "route"],
    "Administration": ["admin", "administration", "office", "fee", "scholarship", "certificate", "principal",
                       "exam", "result", "admission", "helpdesk", "revaluation", "notice"],
    "Teaching / Faculty": ["teacher", "professor", "faculty", "lecture", "lecturer", "teaching", "syllabus",
                           "mentor", "doubt", "assignment"],
    "Infrastructure": ["campus", "building", "lift", "auditorium", "parking", "road", "lighting", "garden",
                       "walkway", "water", "cooler"],
}


def detect_category(text):
    """Return the topic of the feedback, e.g. 'Wi-Fi / Internet'.
    Counts keyword matches for each category and returns the best one.
    If nothing matches, returns 'Other'."""
    text = str(text).lower().replace("wi-fi", "wifi")
    tokens = re.findall(r"[a-z]+", text)

    # Also try the singular form of plural words (washrooms -> washroom, buses -> bus)
    forms = []
    for token in tokens:
        forms.append({token})
        if token.endswith("s"):
            forms[-1].add(token[:-1])
        if token.endswith("es"):
            forms[-1].add(token[:-2])

    best_category, best_score = "Other", 0
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for word_forms in forms if word_forms & set(keywords))
        if score > best_score:  # on a tie, the first category in the list wins
            best_category, best_score = category, score
    return best_category


# ---------------------------------------------------------------------------
# 6. Model evaluation from the command line:   python sentiment.py
#    Compares the ACTUAL labels in feedback.csv with RoBERTa's PREDICTIONS.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os

    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feedback.csv")
    if not os.path.exists(csv_path):
        raise SystemExit("feedback.csv not found.")
    data = pd.read_csv(csv_path, encoding="utf-8-sig")
    if "feedback_text" not in data.columns or "sentiment" not in data.columns:
        raise SystemExit("feedback.csv needs 'feedback_text' and 'sentiment' columns.")

    # Keep only rows that have a valid text and a valid actual label
    data["actual"] = data["sentiment"].astype(str).str.strip().str.capitalize()
    data["clean"] = data["feedback_text"].apply(clean_text)
    data = data[data["actual"].isin(LABELS) & data["clean"].notna()].reset_index(drop=True)
    print(f"Evaluating the pretrained model on {len(data)} labelled feedback entries...")

    tokenizer, model = load_model()
    results = predict_batch(data["clean"].tolist(), tokenizer, model)
    data["predicted"] = [r["label"] for r in results]  # predictions come from the model only

    report = evaluate_predictions(data["actual"].tolist(), data["predicted"].tolist())
    print(f"\nAccuracy        : {report['accuracy'] * 100:.2f}%")
    print(f"Macro Precision : {report['macro_precision']:.3f}")
    print(f"Macro Recall    : {report['macro_recall']:.3f}")
    print(f"Macro F1-score  : {report['macro_f1']:.3f}")
    print("\nPer-class results:")
    print(report["per_class"].to_string(index=False))
    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print(report["confusion_matrix"].to_string())

    wrong = data[data["actual"] != data["predicted"]]
    print(f"\nWrongly predicted: {len(wrong)}. First 10:")
    for _, row in wrong.head(10).iterrows():
        print(f"  actual={row['actual']:<8} predicted={row['predicted']:<8} | {row['feedback_text']}")
