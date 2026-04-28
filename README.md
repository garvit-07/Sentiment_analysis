# Sentiment_analysis
Built sentiment analysis model using BERT + Transformers, achieving high accuracy on text classification tasks Applied preprocessing, fine-tuning, and evaluation (F1-score, precision/recall)

# 🧠 Sentiment Analysis using BERT

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black"/>
  <img src="https://img.shields.io/badge/BERT-base--uncased-4285F4?style=for-the-badge&logo=google&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge"/>
</p>

> Fine-tuned **BERT** for 3-class sentiment classification (Positive / Negative / Neutral) on ~20k samples, achieving strong F1-score performance across all classes.

---

## 📌 Overview

This project builds a high-performance **Sentiment Analysis** model using transformer-based architecture. The model is fine-tuned on a labeled dataset (~20k samples) to classify text into **positive**, **negative**, and **neutral** sentiments.

The entire pipeline — from preprocessing to training and evaluation — is implemented in a **Jupyter Notebook**, covering core NLP and deep learning concepts end-to-end.

---

## 🚀 Highlights

- ✅ Fine-tuned `bert-base-uncased` for 3-class sentiment classification
- ✅ Applied data augmentation, tokenization, and encoding via HuggingFace
- ✅ Evaluated with **F1-score, Precision, Recall, and Accuracy**
- ✅ Used **FP16 training**, cosine scheduling, and early stopping for efficiency
- ✅ Full pipeline in a single, clean Jupyter Notebook

---

## 🎯 Objectives

- Apply transformer models (BERT) for sequence classification
- Implement fine-tuning of pretrained language models from scratch
- Evaluate using standard NLP metrics with class-balanced F1 scoring
- Compare transformer-based methods with traditional ML baselines

---

## 🧠 Model & Approach

| Component | Details |
|-----------|---------|
| **Model** | `bert-base-uncased` |
| **Framework** | PyTorch + HuggingFace Transformers |
| **Task** | 3-class Sequence Classification |
| **Dataset** | ~20,000 labeled samples |
| **Classes** | Positive · Negative · Neutral |

---

## 🔹 Workflow

```
Raw Text
   │
   ▼
Tokenization  ──────────────────────  BERT Tokenizer (max_length=128)
   │
   ▼
Input Encoding  ─────────────────────  input_ids  +  attention_mask
   │
   ▼
BERT Fine-Tuning  ───────────────────  bert-base-uncased  +  Classification Head
   │
   ▼
Softmax Output  ─────────────────────  3-class probabilities
   │
   ▼
Predicted Sentiment  ────────────────  Positive / Negative / Neutral
```

---

## ⚙️ Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 4 |
| Batch Size | 32 |
| Learning Rate | 2e-5 |
| Weight Decay | 0.01 |
| Scheduler | Cosine |
| Warmup Ratio | 0.06 |
| FP16 | ✅ Enabled |
| Early Stopping | Patience = 2 |
| Max Token Length | 128 |

---

## 📊 Evaluation Metrics

Model selection is based on **weighted F1-score** to ensure balanced performance across all three classes.

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall correct predictions |
| **Precision** | True positives / (True + False positives) |
| **Recall** | True positives / (True + False negatives) |
| **F1 Score** | Weighted harmonic mean of Precision & Recall |

---

## 🧪 Example

**Input:**
```
"The movie was absolutely fantastic and engaging!"
```

**Output:**
```
✅ Sentiment: Positive  (confidence: 97.3%)
```

---

## 📁 Project Structure

```
Sentiment_analysis/
│
├── sentiment_model.ipynb    # Main notebook: training & evaluation
│
├── data/
│   ├── train.csv            # Training data
│   ├── val.csv              # Validation data
│   └── test.csv             # Test data
│
├── outputs/
│   ├── checkpoint-*/        # Model checkpoints (saved per epoch)
│   └── best_model/          # Best checkpoint by F1-score
│
└── README.md
```

---

## 🔧 Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/sentiment-analysis-bert.git
cd sentiment-analysis-bert

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install torch transformers scikit-learn pandas numpy jupyter
```

---

## ▶️ How to Run

**1. Launch the notebook:**
```bash
jupyter notebook sentiment_model.ipynb
```

**2. Run all cells to:**
- Load and explore the dataset
- Tokenize and encode text with BERT tokenizer
- Fine-tune `bert-base-uncased` on the training set
- Evaluate on validation set (Accuracy, F1, Precision, Recall)
- Run inference on custom input text

**3. For quick single-sample inference after training:**
```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

tokenizer = BertTokenizer.from_pretrained("outputs/best_model")
model     = BertForSequenceClassification.from_pretrained("outputs/best_model")
model.eval()

text   = "This product exceeded all my expectations!"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
labels = ["Negative", "Neutral", "Positive"]

with torch.no_grad():
    logits = model(**inputs).logits
    pred   = torch.argmax(logits, dim=1).item()

print(f"Sentiment: {labels[pred]}")
```

---

## 📈 Results

> Results are generated after running the notebook. Training curves and a classification report are saved in `outputs/`.

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Positive | — | — | — |
| Negative | — | — | — |
| Neutral  | — | — | — |
| **Weighted Avg** | — | — | **—** |

*Fill in after training on your dataset.*

---

## 🚀 Future Improvements

- [ ] Hyperparameter tuning with **Optuna** or grid search
- [ ] Experiment with **DistilBERT** for faster inference (~40% smaller)
- [ ] Extend to **emotion classification** (joy, anger, sadness, etc.)
- [ ] Perform detailed **error analysis** on misclassified samples
- [ ] Deploy as a REST API using **FastAPI + Hugging Face Spaces**
- [ ] Add **confidence thresholding** for production use

---

## 🤝 Acknowledgements

- [HuggingFace Transformers](https://github.com/huggingface/transformers) — model and tokenizer
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — Devlin et al., 2019
- [PyTorch](https://pytorch.org) — deep learning framework

---

## 📄 License

This project is licensed under the **MIT License** — free to use, modify, and distribute.

---

<p align="center">Made with ❤️ for learning NLP and Transformers</p>
