"""
app.py
------
VoiceOfCampus - ML-Based College Feedback Analyzer
Two sections:  Student Feedback  and  Admin Dashboard

Run with:  streamlit run app.py

feedback.csv columns:
  feedback_text        -> the student's feedback
  sentiment            -> ACTUAL (ground-truth) label, only for our labelled dataset (used for evaluation)
  predicted_sentiment  -> output of the pretrained RoBERTa model
  confidence           -> model confidence in percent
  issue_category       -> topic of the feedback (simple keyword matching)
  timestamp            -> date and time of the feedback
"""

import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

import sentiment  # our own file: model + category + evaluation code

st.set_page_config(page_title="VoiceOfCampus", page_icon="🎓")

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "feedback.csv")
COLUMNS = ["feedback_text", "sentiment", "predicted_sentiment", "confidence", "issue_category", "timestamp"]
LABELS = sentiment.LABELS  # ["Positive", "Neutral", "Negative"]


# ---------------------------------------------------------------------------
# Load the model ONCE (st.cache_resource keeps it in memory between reruns)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading RoBERTa model (first run downloads about 500 MB)...")
def get_model():
    return sentiment.load_model()


try:
    tokenizer, model = get_model()
except Exception as error:
    st.error("Could not load the sentiment model.")
    st.write("Check your internet connection (needed for the first download), "
             "free disk space, and that `torch` and `transformers` are installed.")
    st.code(str(error))
    st.stop()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
page = st.sidebar.radio("Select a section", ["Student Feedback", "Admin Dashboard"])


# ---------------------------------------------------------------------------
# Helper functions: reading / saving the CSV
# ---------------------------------------------------------------------------
def load_data():
    """Read feedback.csv and make sure all needed columns exist."""
    if not os.path.exists(CSV_FILE):
        st.warning("feedback.csv was not found. It will be created when feedback is saved.")
        return pd.DataFrame(columns=COLUMNS)
    try:
        df = pd.read_csv(CSV_FILE, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=COLUMNS)
    except Exception as error:
        st.error(f"Could not read feedback.csv: {error}")
        st.stop()

    if "feedback_text" not in df.columns:
        st.error("feedback.csv must contain a 'feedback_text' column.")
        st.stop()

    for col in COLUMNS:  # add any missing column
        if col not in df.columns:
            df[col] = None
    df = df[COLUMNS].copy()
    for col in ["feedback_text", "sentiment", "predicted_sentiment", "issue_category", "timestamp"]:
        df[col] = df[col].astype(object)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")

    # Work out the issue category for rows that do not have one yet
    missing = df["issue_category"].isna()
    df.loc[missing, "issue_category"] = df.loc[missing, "feedback_text"].apply(sentiment.detect_category)
    return df


def save_data(df):
    """Write the dataframe back to feedback.csv."""
    try:
        df.to_csv(CSV_FILE, index=False, encoding="utf-8")
        return True
    except PermissionError:
        st.error("Cannot write feedback.csv. Close it if it is open in Excel and try again.")
        return False


def submitted_mask(df):
    """True for feedback submitted by students through the app.
    Rows of our labelled dataset have an actual label in 'sentiment' (used only for
    evaluation with `python sentiment.py`), so they are not shown on the dashboard."""
    return df["sentiment"].isna()


def pending_mask(df):
    """True for submitted rows that have valid text but no model prediction yet."""
    valid_text = df["feedback_text"].apply(sentiment.clean_text).notna()
    return df["predicted_sentiment"].isna() & valid_text & submitted_mask(df)


def fill_missing_predictions(df):
    """Run the RoBERTa model on every row that has no prediction, then save."""
    cleaned = df["feedback_text"].apply(sentiment.clean_text)
    pending = df.index[pending_mask(df)]
    progress = st.progress(0.0, text="Running the RoBERTa model...")
    chunk = 64
    for start in range(0, len(pending), chunk):
        rows = pending[start:start + chunk]
        results = sentiment.predict_batch([cleaned[i] for i in rows], tokenizer, model)
        for i, result in zip(rows, results):
            df.at[i, "predicted_sentiment"] = result["label"]
            df.at[i, "confidence"] = result["confidence"]
        progress.progress(min((start + chunk) / len(pending), 1.0))
    progress.empty()
    save_data(df)


# ---------------------------------------------------------------------------
# SECTION 1: Student Feedback
# ---------------------------------------------------------------------------
def page_student():
    st.header("Submit Your Feedback")
    text = st.text_area("Write your feedback about any college-related issue or experience:",
                        height=100, max_chars=1000)

    if st.button("Submit Feedback"):
        cleaned = sentiment.clean_text(text)
        if cleaned is None:
            st.warning("Please write some meaningful feedback (at least 3 characters with letters).")
            return

        with st.spinner("Analyzing your feedback..."):
            result = sentiment.predict_batch([cleaned], tokenizer, model)[0]
        label = result["label"]  # the answer of the RoBERTa model
        category = sentiment.detect_category(cleaned)  # topic of the feedback (keyword matching)

        # 'sentiment' (actual label) stays empty because the true label of new feedback is unknown
        new_row = {"feedback_text": cleaned, "sentiment": None,
                   "predicted_sentiment": label, "confidence": result["confidence"],
                   "issue_category": category,
                   "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        df = load_data()
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        if save_data(df):
            st.success("Thank you! Your feedback has been submitted.")

        message = f"Predicted sentiment: {label}"
        if label == "Positive":
            st.success(message)
        elif label == "Negative":
            st.error(message)
        else:
            st.info(message)
        st.write(f"Issue category: {category}")
        st.write(f"Confidence: {result['confidence']:.1f}%")


# ---------------------------------------------------------------------------
# SECTION 2: Admin Dashboard  (counts, recent feedback table, issues chart)
# ---------------------------------------------------------------------------
def page_admin():
    top_left, top_right = st.columns([4, 1])
    top_left.header("Admin Dashboard")
    top_right.button("🔄 Refresh")  # any click re-reads feedback.csv

    df = load_data()

    # First time only: run the model on feedback that has no prediction yet
    pending = int(pending_mask(df).sum())
    if pending > 0:
        st.info(f"{pending} feedback entries have not been analysed by the model yet. "
                "This is a one-time step.")
        if st.button(f"Analyze {pending} feedback entries"):
            fill_missing_predictions(df)
            st.rerun()

    analysed = df[df["predicted_sentiment"].isin(LABELS) & submitted_mask(df)]
    if analysed.empty:
        st.warning("No feedback submitted yet. Submit feedback from the Student Feedback section.")
        return

    # ---- 1. Counts ----
    counts = analysed["predicted_sentiment"].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Feedback", len(analysed))
    c2.metric("Positive", int(counts.get("Positive", 0)))
    c3.metric("Negative", int(counts.get("Negative", 0)))
    c4.metric("Neutral", int(counts.get("Neutral", 0)))

    # ---- 2. Chart: most reported issues (negative feedback only) ----
    negative = analysed[analysed["predicted_sentiment"] == "Negative"]
    st.subheader("Most Reported Issues (Negative Feedback)")
    if negative.empty:
        st.info("No negative feedback yet.")
    else:
        issue_counts = negative["issue_category"].value_counts()
        chart_data = pd.DataFrame({"Issue": issue_counts.index, "Reports": issue_counts.values})
        fig = px.bar(chart_data, x="Issue", y="Reports", text="Reports",
                     color_discrete_sequence=["#7dc8ff"])
        fig.update_layout(xaxis_title=None)
        st.plotly_chart(fig)  # hover over a bar to see its count

    # ---- 3. Recent feedback table with filters ----
    st.subheader("Recent Feedback")
    f1, f2, f3 = st.columns(3)
    sentiment_filter = f1.selectbox("Sentiment", ["All"] + LABELS)
    issue_filter = f2.selectbox("Issue category", ["All"] + sorted(analysed["issue_category"].unique()))
    row_limit = f3.selectbox("Rows to show", [10, 25, 50, 100])

    table = analysed.copy()
    if sentiment_filter != "All":
        table = table[table["predicted_sentiment"] == sentiment_filter]
    if issue_filter != "All":
        table = table[table["issue_category"] == issue_filter]

    # Newest first
    table["time"] = pd.to_datetime(table["timestamp"], errors="coerce")
    table = table.sort_values("time", ascending=False).head(row_limit)
    # Show the MODEL's prediction (not the dataset's actual label) under the name "sentiment"
    table = table[["timestamp", "feedback_text", "predicted_sentiment", "issue_category"]]
    table = table.rename(columns={"predicted_sentiment": "sentiment"})
    st.dataframe(table,
                 hide_index=True,
                 column_config={"feedback_text": st.column_config.TextColumn(width="large")})


# ---------------------------------------------------------------------------
# Sidebar + page title
# ---------------------------------------------------------------------------
st.title("🎓 VoiceOfCampus")
st.caption("ML-Based College Feedback Analyzer")

if page == "Student Feedback":
    page_student()
else:
    page_admin()
