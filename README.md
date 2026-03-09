# 🤖 Social Bot Detection using Transformer Models

This project implements a **Social Bot Detection System** using **fine-tuned Transformer-based Language Models** to classify social media posts as **Human** or **Bot**.

---

## 📌 Project Motivation

Social bots play a major role in spreading misinformation and automated propaganda.  
This project aims to detect bot-generated content using **state-of-the-art NLP models**.

---

## 🧠 Models Used

- BERT (bert-base-uncased)
- RoBERTa (roberta-base)
- DistilBERT (distilbert-base-uncased)
- XLM-RoBERTa (xlm-roberta-base)

Architecture:
- CLS token embedding
- Feedforward Neural Network
- Sigmoid activation

---

## 🗂️ Project Structure

```
.
├── app.py
├── compare_predict.py
├── predict.py
├── evaluate.py
├── evaluate_models.py
├── plot_confusion_matrix.py
├── train_tweetfake_dataset.py
├── train_fox8_dataset.py
├── requirements.txt
├── Paper/
├── Final paper/
└── README.md
```

---

## 📊 Datasets

### TweetFake Dataset
- Balanced human/bot dataset

### Fox Dataset
- Format: user_id, label, text
- Used for domain generalization

---

## ⚙️ Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 🚀 Training

### Train on TweetFake
```bash
python train_tweetfake_dataset.py
```

### Train on Fox Dataset (1 epoch)
```bash
python train_fox8_dataset.py
```

---

## 🔍 Prediction

Single model:
```bash
python predict.py
```

All models:
```bash
python compare_predict.py
```

---

## 📈 Evaluation

```bash
python evaluate_models.py
```

Metrics:
- Accuracy
- Precision
- Recall
- F1-score

---

## 📊 Confusion Matrix

```bash
python plot_confusion_matrix.py
```

---

## 🖥️ Streamlit Apps

Prediction UI:
```bash
streamlit run app.py
```

Metrics Dashboard:
```bash
streamlit run app_metrics.py
```

---

## 🧪 Training Strategy

- Training performed on Google Colab (GPU)
- Local machine used for inference and demo

---

## 🎓 Academic Notes

- Transformers capture contextual semantics
- F1-score balances precision and recall
- Multiple models enable comparative analysis

---

## 📄 Research Paper
  
Final report in `Final paper/`

---

## 📜 License

Academic and research use only.
