# Sentiment_analysis
Built sentiment analysis model using BERT + Transformers, achieving high accuracy on text classification tasks Applied preprocessing, fine-tuning, and evaluation (F1-score, precision/recall)

Overview

This project focuses on building a high-performance Sentiment Analysis model using transformer-based architectures. The model is fine-tuned on a labeled dataset (~20k samples) to classify text into sentiment categories such as positive, negative, and neutral.

The entire pipeline—from preprocessing to training and evaluation—was implemented in a Jupyter Notebook, emphasizing core NLP and deep learning concepts.

🚀 Project Highlights
Built sentiment analysis model using BERT + Transformers
Achieved strong performance on text classification tasks
Applied data preprocessing, fine-tuning, and evaluation
Evaluated using F1-score, Precision, and Recall
🎯 Objectives
Apply transformer models for text classification
Understand and implement fine-tuning of pretrained language models
Evaluate model performance using standard NLP metrics
Compare transformer-based methods with traditional approaches
🧠 Model & Approach
Model: BERT (bert-base-uncased)
Framework: PyTorch + HuggingFace Transformers
Task: Sequence Classification
🔹 Workflow
Raw Text
   ↓
Tokenization (BERT Tokenizer)
   ↓
Input Encoding (input_ids, attention_mask)
   ↓
BERT Model Fine-Tuning
   ↓
Classification Head (Softmax)
   ↓
Predicted Sentiment
⚙️ Training Configuration
Epochs: 4
Batch Size: 32
Learning Rate: 2e-5
Weight Decay: 0.01
Scheduler: Cosine
Warmup Ratio: 0.06
FP16: Enabled
Early Stopping: Patience = 2
📊 Evaluation Metrics
Accuracy
Precision
Recall
F1 Score (Weighted)
Model selection is based on F1-score to ensure balanced performance across classes.
🧪 Example

Input:

"The movie was absolutely fantastic and engaging!"

Output:

Positive
