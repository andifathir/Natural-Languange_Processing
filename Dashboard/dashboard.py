import streamlit as st
import os
import joblib
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline

# Load and prepare data
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['processed_text', 'Category_Label'])
    return df

@st.cache_data
def train_or_load_model(df, model_path):
    X = df['processed_text']
    y = df['Category_Label']
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    tfidf_vectorizer = TfidfVectorizer()
    X_train = tfidf_vectorizer.fit_transform(X_train_raw)
    X_test = tfidf_vectorizer.transform(X_test_raw)

    if os.path.exists(model_path):
        nb_clf = joblib.load(model_path)
    else:
        nb_clf = MultinomialNB()
        nb_clf.fit(X_train, y_train)
        joblib.dump(nb_clf, model_path)

    return nb_clf, tfidf_vectorizer, X_train, y_train, X_test, y_test, X_test_raw

label_map = {
    0: "Casual Slang",
    1: "Internet Slang",
    2: "Offensive Slang",
    3: "No Slang"
}


def display_prediction_details(model, vectorizer, input_text, label_map):
    st.subheader("🔍 Prediction Results")

    input_vector = vectorizer.transform([input_text])
    prediction = model.predict(input_vector)[0]
    probs = model.predict_proba(input_vector)[0]
    pred_label = label_map.get(prediction, str(prediction))
    confidence = probs[prediction]

    st.markdown(f"### 🧠 Predicted Class: **{pred_label}**")
    st.markdown(f"**Confidence:** {confidence:.2%}")

    st.markdown("### 📊 Class Probabilities")
    prob_df = pd.DataFrame({
        'Class': [label_map[i] for i in range(len(probs))],
        'Probability': probs
    }).sort_values(by='Probability', ascending=False)
    st.bar_chart(prob_df.set_index('Class'))

    # Wrap the LIME explanation inside an expander
    with st.expander("🔍 See Detailed Explanation (LIME)"):
        explain_prediction_with_lime(model, vectorizer, input_text, label_map)


def explain_prediction_with_lime(model, vectorizer, input_text, label_map):
    class_names = [label_map[i] for i in sorted(label_map)]
    pipeline = make_pipeline(vectorizer, model)
    explainer = LimeTextExplainer(class_names=class_names)

    lime_exp = explainer.explain_instance(
        input_text,
        pipeline.predict_proba,
        num_features=8,
        labels=[0, 1, 2, 3]
    )

    label_to_plot = pipeline.predict([input_text])[0]
    word_contributions = dict(lime_exp.as_list(label=label_to_plot))

    st.subheader("🧠 Word Contributions (LIME Explanation)")

    # Use Streamlit columns to restrict width
    col1, col2, col3 = st.columns([2, 1, 1])  # middle column is wider

    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))  # tighter size
        colors = ['green' if v > 0 else 'red' for v in word_contributions.values()]
        ax.barh(list(word_contributions.keys()), word_contributions.values(), color=colors)

        ax.set_xlabel("Contribution", fontsize=9)
        ax.set_title(f"Class: {label_map[label_to_plot]}", fontsize=11)
        ax.tick_params(axis='y', labelsize=8)
        ax.tick_params(axis='x', labelsize=8)
        ax.invert_yaxis()
        plt.tight_layout()

        st.pyplot(fig)


def show_model_info():
    st.subheader("📌 Model Info & Performance Snapshot")
    st.markdown("""
    **Model:** RedditSlang v1.1  
    **Description:** This model classifies input text into four predefined slang categories using a Naive Bayes classifier trained on Reddit-style data.  
    **Classes:**  
    - 🟢 **Casual Slang**: Everyday relaxed informal speech  
    - 💬 **Internet Slang**: Online-specific language like "LOL", "brb"  
    - 🚫 **Offensive Slang**: Toxic, abusive, or profane expressions  
    - ✅ **No Slang**: Neutral or formal language 
    """)
    st.markdown("**Overall Performance:** Accuracy around ~85%")

def show_example_predictions():
    st.subheader("💡 Example Predictions")
    examples = [
        ("OMG, that play was lit 🔥, he totally pwned them!", "Internet Slang"),
        ("Ugh, another camper. So skilled. 🙄", "Offensive Slang"),
        ("Let's meet at 8. Bring the documents.", "No Slang")
    ]
    for text, label in examples:
        st.markdown(f"- **Input:** {text}")
        st.markdown(f"  **Predicted Class:** _{label}_")

def show_limitations():
    st.subheader("⚠️ Caveats & Limitations")
    st.markdown("""
    - Accuracy may drop for very short or ambiguous texts.
    - Sarcasm, irony, and multilingual slang may be misclassified.
    - The model is trained primarily on English Reddit content.
    """)

def show_metrics(clf, X_train, y_train, X_test, y_test):
    st.subheader("📈 Model Evaluation Metrics")
    pred_train = clf.predict(X_train)
    pred_test = clf.predict(X_test)

    st.write(f"**Train Accuracy:** {accuracy_score(y_train, pred_train):.4f}")
    st.write(f"**Test Accuracy:** {accuracy_score(y_test, pred_test):.4f}")

    cm = confusion_matrix(y_test, pred_test)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (Test Data)")
    st.pyplot(fig)

def main():
    st.set_page_config(page_title="Slang Classifier", layout="wide")
    st.title("🧪 Slang Classifier Dashboard")

    file_path = 'Dashboard/balanced_dataset_nlpaug.csv'
    model_path = "naive_bayes_model.pkl"
    label_map = {
        0: "Casual Slang",
        1: "Internet Slang",
        2: "Offensive Slang",
        3: "No Slang"
    }

    df = load_data(file_path)
    nb_clf, tfidf_vectorizer, X_train, y_train, X_test, y_test, X_test_raw = train_or_load_model(df, model_path)

    tab1, tab2, tab3 = st.tabs(["🔍 Classify Text", "📊 Model Info", "📚 Examples & Limitations"])

    with tab1:
        st.markdown("### 🎯 Enter Reddit-style or slang-heavy text below:")
        user_input = st.text_area("Input Text", placeholder="Type or paste your text here...", height=180)
        if st.button("Classify Text"):
            if user_input.strip():
                display_prediction_details(nb_clf, tfidf_vectorizer, user_input, label_map)
            else:
                st.warning("Please enter some text before classifying.")

    with tab2:
        show_model_info()
        show_metrics(nb_clf, X_train, y_train, X_test, y_test)

    with tab3:
        show_example_predictions()
        show_limitations()

if __name__ == "__main__":
    main()
