import streamlit as st
import pickle 
import re
import nltk
import os
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf=TfidfVectorizer(stop_words='english')

def ensure_nltk_resource(resource, path):
    try:
        nltk.data.find(path)
        print(f"{resource} already installed")
    except LookupError:
        print(f"{resource} not found, downloading...")
        nltk.download(resource)

ensure_nltk_resource("punkt", "tokenizers/punkt")
ensure_nltk_resource("stopwords", "corpora/stopwords")

# loading the models

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "models")

knn = pickle.load(
    open(os.path.join(MODEL_DIR, "knn_model.pkl"), "rb")
)

tfidf= pickle.load(
    open(os.path.join(MODEL_DIR, "tfidf.pkl"), "rb")
)

le = pickle.load(
    open(os.path.join(MODEL_DIR, "le.pkl"), "rb")
)


# web app

import re

def clean_text(text):
    clean_Txt = re.sub(r'https?://\S+', '', text).strip()
    clean_Txt = re.sub(r'#\w+', '', clean_Txt)
    clean_Txt = re.sub(r'@\w+', '', clean_Txt)
    clean_Txt = re.sub('RT|cc', '', clean_Txt)
    clean_Txt = re.sub(r'[^a-zA-Z0-9]', ' ', clean_Txt)
    clean_Txt = re.sub(r'\s+', ' ', clean_Txt)
    return clean_Txt



def main():
    st.title("Resume Screening App")
    uploaded_file=st.file_uploader("Upload Resume",type=['txt','pdf','docx'])

    if uploaded_file is not None:
        try:
            resume_bytes=uploaded_file.read()  # it read the bytes not the text hence bytes
            resume_text=resume_bytes.decode('utf-8')  # bytes to text
        except UnicodeDecodeError:
        # if unicode fails try laten-1
            resume_text=resume_bytes.decode('latin-1')

        clean_resume=clean_text(resume_text)
        input_features=tfidf.transform([clean_resume])

        prediction_key=knn.predict(input_features)[0]

        st.write("Predicted:",le.inverse_transform([prediction_key])[0])


if __name__=="__main__":
    main()
