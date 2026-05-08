# Phishing URL Detection

A comparative study of machine-learning, deep-learning, and NLP-based approaches for classifying URLs as **Benign**, **Defacement**, **Phishing**, or **Malware**, with the best-performing model deployed behind a Streamlit UI.

## Final model

After comparing feature-based, NLP-based, and deep-learning models, the deployed model is:

- **LinearSVC** trained on **TF-IDF character n-grams (3–5)**
- ~96% accuracy, 0.97 weighted F1 on held-out test data
- Saved as [`final_phishing_svm_model.pkl`](final_phishing_svm_model.pkl) + [`final_tfidf_vectorizer.pkl`](final_tfidf_vectorizer.pkl)

## Repo layout

| File | Purpose |
|---|---|
| [`phishing_urls.ipynb`](phishing_urls.ipynb) | Main notebook — full pipeline, model comparison, final model export |
| [`NLP-BASED_ML.ipynb`](NLP-BASED_ML.ipynb) | NLP/TF-IDF experiments (RF, SVM) |
| [`DL.ipynb`](DL.ipynb) | Deep-learning experiments (CNN, LSTM, BiLSTM) |
| [`ensemble.ipynb`](ensemble.ipynb) | Ensemble experiments |
| [`model.ipynb`](model.ipynb) | Initial model exploration |
| [`app.py`](app.py) | Streamlit UI for the deployed model |
| `final_phishing_svm_model.pkl` | Deployed LinearSVC classifier |
| `final_tfidf_vectorizer.pkl` | Deployed TF-IDF vectorizer |
| `XGboost_model.pkl`, `XGB_vectorizer.pkl` | XGBoost variant from the study |

> The training datasets (`final_dataset_with_all_features_v3.1.csv`, `phishing_urls_nlp_dataset.csv`) and intermediate TF-IDF matrices (`*.npz`) are excluded from the repo — they're either too large for GitHub or trivially regenerable from the notebook.

## Running the UI

```bash
pip install streamlit scikit-learn joblib pandas
streamlit run app.py
```

Opens at `http://localhost:8501`. You can:
- Classify a single URL with a colored verdict card and decision-margin score
- Paste many URLs (one per line) in the **Batch** tab and download results as CSV

## Models compared

| Model | Type | Accuracy | Recall | F1 |
|---|---|---|---|---|
| Random Forest (features) | ML | 0.92 | 0.97 | 0.95 |
| LinearSVC (features) | ML | 0.85 | 0.99 | 0.91 |
| TF-IDF + RF | NLP | 0.74 | 1.00 | 0.84 |
| **TF-IDF + LinearSVC** ✅ | **NLP** | **0.96** | **0.98** | **0.97** |
| CNN | DL | 0.94 | 0.97 | 0.95 |
| LSTM | DL | 0.85 | 0.80 | 0.88 |
| BiLSTM | DL | 0.86 | 0.81 | 0.89 |

## Class labels

| ID | Class |
|---|---|
| 0 | Benign |
| 1 | Defacement |
| 2 | Phishing |
| 3 | Malware |
