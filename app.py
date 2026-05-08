import re
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "final_phishing_svm_model.pkl"
VECTORIZER_PATH = "final_tfidf_vectorizer.pkl"

LABELS = {
    0: ("Benign", "#16a34a"),
    1: ("Defacement", "#ea580c"),
    2: ("Phishing", "#dc2626"),
    3: ("Malware", "#b91c1c"),
}


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def clean_url(url: str) -> str:
    url = str(url).lower().strip()
    url = re.sub(r"https?://", "", url)
    url = re.sub(r"www\.", "", url)
    return url


def predict(urls, model, vectorizer):
    cleaned = [clean_url(u) for u in urls]
    vecs = vectorizer.transform(cleaned)
    preds = model.predict(vecs)
    scores = model.decision_function(vecs)
    confidences = []
    for row in scores:
        if hasattr(row, "__len__"):
            top = sorted(row, reverse=True)
            margin = top[0] - top[1]
        else:
            margin = abs(float(row))
        confidences.append(margin)
    return cleaned, preds, confidences


st.set_page_config(
    page_title="Phishing URL Detector",
    page_icon="🛡️",
    layout="centered",
)

st.title("Phishing URL Detector")
st.caption(
    "LinearSVC + TF-IDF char n-grams (3–5). Trained on the final phishing URL dataset."
)

model, vectorizer = load_artifacts()

tab_single, tab_batch = st.tabs(["Single URL", "Batch (multiple URLs)"])

with tab_single:
    url = st.text_input(
        "Enter a URL to classify",
        placeholder="e.g. http://secure-login-paypal.example.com/verify",
    )
    if st.button("Classify", type="primary", use_container_width=True):
        if not url.strip():
            st.warning("Please enter a URL.")
        else:
            cleaned, preds, confs = predict([url], model, vectorizer)
            label_id = int(preds[0])
            name, color = LABELS.get(label_id, (f"Class {label_id}", "#6b7280"))

            st.markdown(
                f"""
                <div style="
                    padding: 18px 22px;
                    background: {color};
                    color: white;
                    border-radius: 10px;
                    font-size: 1.4rem;
                    font-weight: 600;
                    text-align: center;
                    margin-top: 12px;
                ">
                    {name}
                </div>
                """,
                unsafe_allow_html=True,
            )

            col1, col2 = st.columns(2)
            col1.metric("Predicted class", name)
            col2.metric("Decision margin", f"{confs[0]:.3f}")

            with st.expander("Details"):
                st.write({"original": url, "cleaned": cleaned[0], "label_id": label_id})

with tab_batch:
    st.write("Enter one URL per line:")
    text = st.text_area(
        "URLs",
        height=200,
        placeholder="http://example.com\nhttp://suspicious-site.tk/login\n...",
    )
    if st.button("Classify all", type="primary", use_container_width=True):
        urls = [u.strip() for u in text.splitlines() if u.strip()]
        if not urls:
            st.warning("Please enter at least one URL.")
        else:
            cleaned, preds, confs = predict(urls, model, vectorizer)
            df = pd.DataFrame(
                {
                    "URL": urls,
                    "Cleaned": cleaned,
                    "Prediction": [LABELS.get(int(p), (f"Class {p}", ""))[0] for p in preds],
                    "Decision margin": [round(c, 3) for c in confs],
                }
            )
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download results as CSV",
                df.to_csv(index=False).encode("utf-8"),
                file_name="phishing_predictions.csv",
                mime="text/csv",
            )

st.divider()
st.caption(
    "Classes: 0 Benign · 1 Defacement · 2 Phishing · 3 Malware. "
    "Decision margin = gap between top-2 class scores (higher = more confident)."
)
