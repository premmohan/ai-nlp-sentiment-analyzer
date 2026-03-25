import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Page config
st.set_page_config(
    page_title="Tweet Sentiment Analyzer",
    page_icon="💬",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
.main {
    background-color: #0E1117;
    color: white;
}
.stTextArea textarea {
    border-radius: 10px;
    padding: 10px;
}
.big-font {
    font-size: 20px !important;
}
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    model_path = "bert_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

tokenizer, model = load_model()

labels = ["Negative 😡", "Neutral 😐", "Positive 😊"]

# Header
st.markdown("<h1 style='text-align: center;'>💬 Tweet Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Powered by BERT 🤖</p>", unsafe_allow_html=True)

# Input box
text = st.text_area("✍️ Enter your tweet here:", height=150)

# Predict button
if st.button("🔍 Analyze Sentiment"):

    if text.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        with st.spinner("Analyzing..."):

            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

            with torch.no_grad():
                outputs = model(**inputs)

            logits = outputs.logits
            probs = F.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()

        # Result
        st.markdown("## 🎯 Prediction Result")
        st.success(labels[pred])

        # Confidence scores
        st.markdown("### 📊 Confidence Scores")
        st.progress(float(probs[0][pred]))

        st.write({
            "Negative": float(probs[0][0]),
            "Neutral": float(probs[0][1]),
            "Positive": float(probs[0][2]),
        })

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center;'>Built by PREM MOHAN</p>",
    unsafe_allow_html=True
)