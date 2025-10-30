# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from transformers import pipeline
from wordcloud import WordCloud
import math
import time
import numpy as np

# --- Page config and styling ---
st.set_page_config(page_title="Sentiment Analyzer Dashboard", layout="wide")
st.markdown(
    """
    <style>
    body { background-color: #0E1117; color: #E5E5E5; }
    .stSidebar { background-color: #0E1117; color: #00FF00; }
    h1, h2, h3 { color: #00FF00; }
    </style>
    """, unsafe_allow_html=True
)
st.title("🧠 Sentiment Analyzer Dashboard")
st.markdown("*Analyze emotions, visualize insights, and explore text trends.*")

# --- Model loader (cached) ---
@st.cache_resource
def load_hf_pipeline(model_name="distilbert-base-uncased-finetuned-sst-2-english"):
    return pipeline("sentiment-analysis", model=model_name, device=-1)

# --- Sidebar settings ---
st.sidebar.header("Settings")
model_choice = st.sidebar.selectbox("Choose sentiment engine", ["Hugging Face (transformers)", "TextBlob (simple)"])
hf_model_name = st.sidebar.text_input("HF model name", value="distilbert-base-uncased-finetuned-sst-2-english")
batch_size = st.sidebar.number_input("Batch size for CSV", min_value=8, max_value=256, value=32)

if model_choice.startswith("Hugging"):
    with st.spinner("Loading Hugging Face model..."):
        hf_pipeline = load_hf_pipeline(hf_model_name)
    st.sidebar.success(f"Loaded: {hf_model_name}")

# --- Single text analysis ---
st.subheader("🔹 Single Text Analysis")
text_input = st.text_area("Enter text to analyze", value="I love Streamlit — it's super easy to use!")

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

if st.button("Analyze Text"):
    if not text_input.strip():
        st.warning("Please enter some text first.")
    else:
        result = analyze_single(text_input)
        if result["polarity"] > 0:
            st.success(f"🙂 Positive — Score: {result['score']:.3f}")
        elif result["polarity"] < 0:
            st.error(f"🙁 Negative — Score: {result['score']:.3f}")
        else:
            st.info(f"😐 Neutral — Score: {result['score']:.3f}")
        st.write(result)

# --- CSV upload / batch analysis ---
st.subheader("📁 Batch Sentiment Analysis (CSV)")
st.markdown("Upload a CSV with a column named `text`.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if "text" not in df.columns:
        st.error("CSV must contain a column named `text`.")
    else:
        texts = df["text"].astype(str).tolist()
        n = len(texts)
        st.write(f"Uploaded {n} rows.")

        if st.button("Run batch sentiment analysis"):
            results = []
            progress = st.progress(0)
            num_batches = math.ceil(n / batch_size)

            for i in range(num_batches):
                start = i * batch_size
                end = min(start + batch_size, n)
                batch_texts = texts[start:end]

                if model_choice.startswith("Hugging"):
                    batch_out = hf_pipeline(batch_texts)
                    for o in batch_out:
                        label = o["label"]
                        score = o["score"]
                        polarity = score if label == "POSITIVE" else -score
                        results.append({"label": label, "score": score, "polarity": polarity})
                else:
                    for t in batch_texts:
                        blob = TextBlob(t)
                        polarity = blob.sentiment.polarity
                        label = "POSITIVE" if polarity > 0 else ("NEGATIVE" if polarity < 0 else "NEUTRAL")
                        results.append({"label": label, "score": abs(polarity), "polarity": polarity})

                progress.progress(int((i + 1) / num_batches * 100))
                time.sleep(0.01)

            res_df = pd.DataFrame(results)
            out_df = pd.concat([df.reset_index(drop=True), res_df.reset_index(drop=True)], axis=1)
            st.write("### 🎯 Results")
            st.dataframe(out_df)

            # --- Side-by-side layout ---
            col1, col2 = st.columns(2)

            # Left: Sentiment Pie Chart
            with col1:
                st.write("### 📊 Sentiment Distribution")
                counts = out_df["label"].value_counts()
                fig, ax = plt.subplots()
                ax.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90)
                ax.axis("equal")
                st.pyplot(fig)

            # Right: Sentiment-aware Word Cloud
            with col2:
                st.write("### ☁️ Sentiment Word Cloud")
                word_scores = {}
                for text, polarity in zip(out_df["text"].astype(str), out_df["polarity"]):
                    for word in text.split():
                        word = word.lower()
                        if word not in word_scores:
                            word_scores[word] = []
                        word_scores[word].append(polarity)
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
                wc = WordCloud(width=400, height=300, background_color="black",
                               color_func=sentiment_color).generate(all_text)
                fig2, ax2 = plt.subplots()
                ax2.imshow(wc, interpolation="bilinear")
                ax2.axis("off")
                st.pyplot(fig2)

            # Download CSV
            csv_bytes = out_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download results CSV", csv_bytes, file_name="sentiment_results.csv", mime="text/csv")

st.markdown("---")
st.markdown("**Powered by Transformers & TextBlob — accuracy may vary by context.**")

