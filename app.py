import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from transformers import pipeline
from typing import List
import math
import time

st.set_page_config(page_title="Sentiment Analyzer (HF + TextBlob)", layout="wide")
st.title("🧠 Sentiment Analyzer Dashboard — Transformers + TextBlob")

@st.cache_resource
def load_hf_pipeline(model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
    return pipeline("sentiment-analysis", model=model_name, device=-1)

st.sidebar.header("Settings")
model_choice = st.sidebar.selectbox("Choose sentiment engine", ["Hugging Face (transformers)", "TextBlob (simple)"])
hf_model_name = st.sidebar.text_input("HF model name", value="distilbert-base-uncased-finetuned-sst-2-english")
batch_size = st.sidebar.number_input("Batch size for CSV", min_value=8, max_value=256, value=32)

if model_choice.startswith("Hugging"):
    with st.spinner("Loading Hugging Face model..."):
        hf_pipeline = load_hf_pipeline(hf_model_name)
    st.sidebar.success(f"Loaded: {hf_model_name}")

st.subheader("🔹 Single Text Analysis")
text_input = st.text_area("Enter text to analyze", value="I love Streamlit — it's super easy to use!")

def analyze_single(text: str):
    if model_choice.startswith("Hugging"):
        out = hf_pipeline(text[:1000])[0]
        label = out["label"]
        score = out["score"]
        polarity = score if label == "POSITIVE" else -score
        return {"label": label, "score": score, "polarity": polarity}
    else:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        if polarity > 0:
            label = "POSITIVE"
        elif polarity < 0:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"
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

            st.write("### 📊 Sentiment Distribution")
            counts = out_df["label"].value_counts()
            fig, ax = plt.subplots()
            ax.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90)
            ax.axis("equal")
            st.pyplot(fig)

            csv_bytes = out_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download results CSV", csv_bytes, file_name="sentiment_results.csv", mime="text/csv")

st.markdown("---")
st.markdown("**Tips:** Use the Hugging Face model for better accuracy. For very large datasets, run batch jobs offline.")
