
# 🔍 Product Outage Classifier

This repository contains a machine learning pipeline designed to classify social media posts that reference **product outages** at Starbucks. The project leverages both traditional NLP techniques and modern deep learning embeddings to build a high-performing classifier capable of identifying rare but impactful customer feedback.

---

## 🎯 Project Objective

Starbucks aims to supplement traditional inventory tracking with insights from customer posts on platforms like **Facebook**, **Instagram**, and **Uber Eats**. This classifier identifies posts related to product outages, helping prioritize inventory decisions and improve operational responsiveness.

---

## 🧪 Methodology Overview

### 📊 Data Sources
- **Facebook & Instagram**: 76,037 posts
- **Uber Eats**: 84,055 reviews
- Posts were manually labeled for outage relevance (binary classification: 1 = outage, 0 = not outage)

### 🧼 Preprocessing
- Removed nulls and duplicates
- Trimmed long posts to 1,000 characters
- Stratified sampling to balance training data (40% outage-like posts)

### ⚙️ Feature Engineering
Three approaches were tested:
1. **TF-IDF** with Gensim Phraser for phrase detection
2. **BERT embeddings** using `bert-base-uncased`
3. **Combined TF-IDF + BERT** for hybrid semantic + frequency representation

### 🧠 Model Training
Four classifiers were evaluated:
- Logistic Regression
- Support Vector Machine (SVM)
- XGBoost
- LightGBM

Each model was trained using 5-fold stratified cross-validation and tuned for optimal F1 score.

### 🏆 Best Model
- **LightGBM** with **combined TF-IDF + BERT features**
- Final test set performance:
  - **F1 Score**: 0.6316
  - **Precision**: 0.8571
  - **Recall**: 0.5000
  - **Log Loss**: 0.0348

---

## 🚀 Deployment

The final model is deployed via a **Streamlit web app**, allowing users to input text and receive real-time predictions on whether the post references a product outage.

---

## 📁 Repository Structure

├── Project_Code.pdf # Annotated code and results 

├── Project_Overview.pdf # Executive summary and methodology 

├── README.md # Repository documentation

---

## 🛠 Tech Stack

- **Python**: Core language
- **Pandas / NumPy / Scikit-learn**: Data manipulation and modeling
- **Gensim**: Phrase detection
- **Transformers (Hugging Face)**: BERT embeddings
- **LightGBM / XGBoost / SVM / Logistic Regression**: Classifiers
- **Streamlit**: Web app deployment

---

## 📌 Key Insights

- Combining **semantic** and **frequency-based** features improves classification of nuanced language.
- **LightGBM** excels in handling imbalanced data and complex feature interactions.
- The model can serve as a **real-time alert system** for inventory teams, flagging customer-reported outages.

---

## 📈 Future Work

- Expand labeled dataset to improve recall
- Explore active learning at scale
- Integrate model into real-time monitoring pipelines
