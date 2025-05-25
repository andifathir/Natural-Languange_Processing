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

# Predict and show metrics (optional, can be extended)
def show_metrics(clf, X_train, y_train, X_test, y_test):
    st.subheader("Model Performance")

    pred_train = clf.predict(X_train)
    pred_test = clf.predict(X_test)

    st.write(f"Train Accuracy: {accuracy_score(y_train, pred_train):.4f}")
    st.write(f"Test Accuracy: {accuracy_score(y_test, pred_test):.4f}")

    cm = confusion_matrix(y_test, pred_test)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (Test Data)")
    st.pyplot(fig)

def main():
    st.title("NLP Text Classification Dashboard")

    file_path = 'Dashboard/balanced_dataset_nlpaug.csv'
    model_path = "naive_bayes_model.pkl"

    df = load_data(file_path)
    nb_clf, tfidf_vectorizer, X_train, y_train, X_test, y_test, X_test_raw = train_or_load_model(df, model_path)

    show_metrics(nb_clf, X_train, y_train, X_test, y_test)

    st.write("---")
    st.subheader("Classify Your Own Text")

    user_input = st.text_area("Enter text to classify", height=150)

    if user_input:
        user_tfidf = tfidf_vectorizer.transform([user_input])
        prediction = nb_clf.predict(user_tfidf)[0]

        # Map numeric label back to original class name if you want
        label_map = {
            0: "Casual Slang",
            1: "Internet Slang",
            2: "Offensive Slang",
            3: "No Slang"
        }
        pred_label = label_map.get(prediction, str(prediction))

        st.markdown(f"### Prediction: **{pred_label}**")

if __name__ == "__main__":
    main()
