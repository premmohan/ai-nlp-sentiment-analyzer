# 🐦 Sentiment Analysis using BERT

This project is a Twitter Sentiment Analysis web application built using **BERT (Bidirectional Encoder Representations from Transformers)** and deployed with **Streamlit**.

The model classifies tweets into three categories:

- ✅ Positive  
- 😐 Neutral  
- ❌ Negative  

---

## 📂 Project Structure

Project_BERT/

│── app.py              # Streamlit web application  
│── bert_model/         # Fine-tuned BERT model folder  
│── requirements.txt    # Project dependencies  
│── README.md           # Project documentation  

---

## 🧠 Model Details

- Pretrained BERT model from HuggingFace Transformers
- Fine-tuned on a Twitter sentiment dataset
- 3-class sentiment classification
- Tokenization handled using BERT tokenizer
- Outputs predicted sentiment label
