# app.py
import streamlit as st
import pandas as pd
import numpy as np
from textblob import TextBlob
from transformers import pipeline
from wordcloud import WordCloud
import plotly.express as px
import time

# --- Page config ---
st.set_page_config(
    page_title="Sentiment Analyzer",
    layout="wide",
    page_icon="🧠"
)

# --- Custom CSS ---
st.markdown("""
    <style>
    body { background-color: #0E1117; color: #E5E5E5; }
    .stSidebar { background-color: #0E1117; color: #00FF00; }
    h1, h2, h3 { color: #00FF00; }
    .stButton>button { background-color: #00FF00; color: #0E1117; }
    .stDownloadButton>button { background-color: #00FF00; color: #0E1117; }
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.title("🧠 Sentiment Analyzer Dashboard")
st.markdown("*Analyze emotions, visualize insights, and explore text trends.*")

# --- Sidebar ---
st.sidebar.header("Settings")
model_choice = st.sidebar.selectbox("Choose sentiment engine", ["Hugging Face (transformers)", "TextBlob"])
hf_model_name = st.sidebar.text_input("HF model name", value="distilbert-base-uncased-finetuned-sst-2-english")
batch_size = st.sidebar.number_input("Batch size for CSV", min_value=8, max_value=256, value=32)

@st.cache_resource
def load_hf(model_name):
    return pipeline("sentiment-analysis", model=model_name, device=-1)

if model_choice.startswith("Hugging"):
    with st.spinner("Loading Hugging Face model..."):
        hf_pipeline = load_hf(hf_model_name)
    st.sidebar.success(f"Loaded: {hf_model_name}")

# --- Functions ---
def analyze_single(text):
    if model_choice.startswith("Hugging"):
        out = hf_pipeline(text[:1000])[0]
        label = out["label"]
        score = out["score"]
        polarity = score if label == "POSITIVE" else -score
        return {"label": label, "score": score, "polarity": polarity}
    else:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        label = "POSITIVE" if polarity > 0 else ("NEGATIVE" if polarity < 0 else "NEUTRAL")
        return {"label": label, "score": abs(polarity), "polarity": polarity}

# --- Single text analysis ---
st.subheader("🔹 Single Text Analysis")
text_input = st.text_area("Enter text here:")

if st.button("Analyze Text"):
    if not text_input.strip():
        st.warning("Please enter some text first.")
    else:
        result = analyze_single(text_input)
        st.markdown(f"**Label:** {result['label']}  |  **Score:** {result['score']:.3f}")
        st.markdown(f"**Polarity:** {result['polarity']:.3f}")

# --- CSV batch analysis ---
st.subheader("📁 Batch Sentiment Analysis (CSV)")
st.markdown("Upload a CSV with a column named `text`.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if "text" not in df.columns:
        st.error("CSV must contain a 'text' column.")
    else:
        texts = df["text"].astype(str).tolist()
        n = len(texts)
        st.write(f"Uploaded {n} rows.")

        if st.button("Run Batch Analysis"):
            results = []
            progress = st.progress(0)
            num_batches = int(np.ceil(n / batch_size))

            for i in range(num_batches):
                start = i * batch_size
                end = min(start + batch_size, n)
                batch_texts = texts[start:end]

                for t in batch_texts:
                    res = analyze_single(t)
                    results.append(res)

                progress.progress(int((i + 1)/num_batches*100))
                time.sleep(0.01)

            out_df = pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)
            st.write("### 🎯 Results")
            st.dataframe(out_df)

            # --- Side-by-side layout ---
            col1, col2 = st.columns(2)

            # Left: Interactive Pie Chart
            with col1:
                counts = out_df["label"].value_counts()
                fig = px.pie(
                    names=counts.index,
                    values=counts.values,
                    color=counts.index,
                    color_discrete_map={"POSITIVE":"green", "NEGATIVE":"red", "NEUTRAL":"grey"},
                    title="Sentiment Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)

            # Right: Sentiment-aware Word Cloud
            with col2:
                word_scores = {}
                for text, polarity in zip(out_df["text"].astype(str), out_df["polarity"]):
                    for word in text.split():
                        word = word.lower()
                        word_scores.setdefault(word, []).append(polarity)
                word_avg = {w: np.mean(scores) for w, scores in word_scores.items()}

                def sentiment_color(word, **kwargs):
                    score = word_avg.get(word.lower(), 0)
                    if score > 0.05:
                        return "green"
                    elif score < -0.05:
                        return "red"
                    else:
                        return "grey"

                all_text = " ".join(out_df["text"].astype(str))
                wc = WordCloud(width=500, height=400, background_color="black", color_func=sentiment_color).generate(all_text)
                st.image(wc.to_array(), use_column_width=True)

            # Download CSV
            csv_bytes = out_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Results CSV", csv_bytes, file_name="sentiment_results.csv", mime="text/csv")

st.markdown("---")
st.markdown("**Powered by Transformers & TextBlob — accuracy may vary by context.**")


