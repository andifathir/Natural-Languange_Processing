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
def train_or_load_model(model_path, file_path=None):
    if os.path.exists(model_path):
        loaded = joblib.load(model_path)
        if isinstance(loaded, tuple) and len(loaded) == 2:
            nb_clf, tfidf_vectorizer = loaded
            return nb_clf, tfidf_vectorizer, None, None, None, None, None
        else:
            raise ValueError("Saved model file is not a (model, vectorizer) tuple.")
    else:
        if file_path is None:
            raise ValueError("Model not found and no dataset provided to train it.")

        df = pd.read_csv(file_path).dropna(subset=['processed_text', 'Category_Label'])

        X = df['processed_text']
        y = df['Category_Label']
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        tfidf_vectorizer = TfidfVectorizer()
        X_train = tfidf_vectorizer.fit_transform(X_train_raw)
        X_test = tfidf_vectorizer.transform(X_test_raw)

        nb_clf = MultinomialNB()
        nb_clf.fit(X_train, y_train)
        joblib.dump((nb_clf, tfidf_vectorizer), model_path)

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

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
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
    st.markdown("**Overall Performance:** Accuracy around ~80%")

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

def run_batch_prediction(model, vectorizer, label_map):
    st.subheader("📂 Batch Slang Classification")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if 'processed_text' not in df.columns:
            st.error("CSV must contain a 'processed_text' column.")
            return

        X = df['processed_text'].fillna("")
        X_vectorized = vectorizer.transform(X)
        predictions = model.predict(X_vectorized)
        probs = model.predict_proba(X_vectorized)

        df['Predicted_Label'] = predictions
        df['Predicted_Class'] = df['Predicted_Label'].map(label_map)
        df['Max_Probability'] = probs.max(axis=1)

        if 'Category_Label' in df.columns:
            correct = (df['Predicted_Label'] == df['Category_Label']).sum()
            total = len(df)
            accuracy = correct / total
            st.success(f"✅ Correct Predictions: {correct} / {total} ({accuracy:.2%})")
            st.error(f"❌ Incorrect Predictions: {total - correct} / {total} ({1 - accuracy:.2%})")
        else:
            st.warning("⚠️ Ground truth (`Category_Label`) not found. Accuracy can't be computed.")

        st.markdown("### 📄 Preview of Results")

        if 'Category_Label' in df.columns:
            df['Actual_Class'] = df['Category_Label'].map(label_map)
            preview_cols = ['processed_text', 'Actual_Class', 'Predicted_Class', 'Max_Probability']
        else:
            preview_cols = ['processed_text', 'Predicted_Class', 'Max_Probability']

        st.dataframe(df[preview_cols].head())

        with st.expander("🔍 Show Full Result Table"):
            st.data_editor(df[preview_cols], use_container_width=True, num_rows="dynamic")

        st.markdown("### 📊 Predicted Class Distribution")

        class_counts = df['Predicted_Class'].value_counts().reindex(label_map.values(), fill_value=0)

        dist_df = pd.DataFrame({
            'Predicted Count': class_counts.values
        }, index=class_counts.index)

        st.bar_chart(dist_df)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Predictions CSV", data=csv, file_name="batch_predictions.csv", mime="text/csv")

def main():
    st.set_page_config(page_title="Slang Classifier", layout="wide")
    st.title("🧪 Slang Classifier Dashboard")

    file_path = 'preprocessed_augmented_dataasdasasdasddasd.csv'
    model_path = "naive_bayes_model.pkl"

    # Only pass file_path if training is needed
    nb_clf, tfidf_vectorizer, X_train, y_train, X_test, y_test, X_test_raw = train_or_load_model(
        model_path=model_path,
        file_path=file_path if not os.path.exists(model_path) else None
    )

    df = load_data(file_path) if os.path.exists(file_path) else None

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Classify Text",
        "📂 Batch Prediction",
        "📈 Model Info",
        "📚 Examples & Limitations"
    ])

    with tab1:
        st.markdown("### 🎯 Enter Reddit-style or slang-heavy text below:")
        user_input = st.text_area("Input Text", placeholder="Type or paste your text here...", height=180)
        if st.button("Classify Text"):
            if user_input.strip():
                display_prediction_details(nb_clf, tfidf_vectorizer, user_input, label_map)
            else:
                st.warning("Please enter some text before classifying.")

    with tab2:
        run_batch_prediction(nb_clf, tfidf_vectorizer, label_map)

    with tab3:
          show_model_info()

    with tab4:
        show_example_predictions()
        show_limitations()

if __name__ == "__main__":
    main()
