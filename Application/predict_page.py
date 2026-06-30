import streamlit as st
import numpy as np
import re
import pickle
import nltk
from nltk.stem import PorterStemmer

# ======================== Safe NLTK setup ========================
def setup_nltk():
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")
    from nltk.corpus import stopwords
    return set(stopwords.words("english"))

stopwords = setup_nltk()

# ======================== Load Saved Models ========================
lg = pickle.load(open('logistic_regresion.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
lb = pickle.load(open('label_encoder.pkl', 'rb'))

# ======================== Helper Functions ========================
def clean_text(text):
    stemmer = PorterStemmer()
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower().split()
    text = [stemmer.stem(word) for word in text if word not in stopwords]
    return " ".join(text)

def predict_emotion(input_text):
    cleaned_text = clean_text(input_text)
    input_vectorized = tfidf_vectorizer.transform([cleaned_text])
    predicted_label = lg.predict(input_vectorized)[0]
    predicted_emotion = lb.inverse_transform([predicted_label])[0]
    probability = np.max(lg.predict_proba(input_vectorized))
    return predicted_emotion, probability

# ======================== Streamlit Page Function ========================
def show_predict_page():
    st.set_page_config(page_title="Emotion Recognition System", page_icon="🧠", layout="wide")

    # --- Force Scroll to Top ---
    st.markdown("""
        <script>
            window.addEventListener('load', function() {
                window.scrollTo({top: 0, behavior: 'smooth'});
            });
        </script>
    """, unsafe_allow_html=True)

    # --- Custom CSS (Home Page Matching) ---
    st.markdown("""
        <style>
        @keyframes gradientFlow {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }
        .main {
            background: linear-gradient(120deg, #e3f2fd, #e8f5e9, #fff3e0);
            background-size: 300% 300%;
            animation: gradientFlow 10s ease infinite;
            font-family: 'Poppins', sans-serif;
            color: #333;
        }
        .header-bar {
            width: 100%;
            background: linear-gradient(90deg, #b71c1c, #d32f2f, #f44336);
            color: white;
            padding: 25px 0;
            text-align: center;
            border-radius: 0 0 20px 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .header-title {
            font-size: 42px;
            font-weight: 700;
            margin-bottom: 5px;
            letter-spacing: 1px;
        }
        .header-subtitle {
            font-size: 18px;
            font-weight: 400;
            color: #fff8f8;
        }
        h1, h2 {
            text-align: center;
            color: #b71c1c;
            font-weight: 700;
        }
        h1 {
            font-size: 52px;
            margin-bottom: 0px;
        }
        .divider {
            height: 3px;
            width: 200px;
            background: linear-gradient(to right, #e53935, #ef5350);
            border-radius: 5px;
            margin: 15px auto 30px;
        }
        .glass {
            background: rgba(255,255,255,0.75);
            backdrop-filter: blur(12px);
            border-radius: 25px;
            padding: 30px 40px;
            margin: 25px auto;
            width: 80%;
            box-shadow: 0 6px 25px rgba(0,0,0,0.15);
        }
        .intro {
            text-align: justify;
            font-size: 18px;
            color: #222;
            line-height: 1.8;
        }
        .input-box {
            background-color: #ffffff;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 0 15px rgba(0,0,0,0.2);
            margin: 20px auto;
            width: 80%;
        }
        .predict-card {
            background: rgba(255,255,255,0.9);
            backdrop-filter: blur(8px);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 0 20px rgba(0,0,0,0.25);
            margin: 30px auto;
            width: 60%;
        }
        .emotion {
            font-size: 28px;
            font-weight: 700;
            color: #1b5e20;
        }
        .probability {
            font-size: 20px;
            color: #ff6f00;
        }
        </style>
    """, unsafe_allow_html=True)

    # --- Header ---
    st.markdown("""
        <div class='header-bar'>
            <div class='header-title'>🎓 Emotion Recognition Using Deep Learning</div>
            <div class='header-subtitle'>Final Year Project • Department of Computer Science</div>
        </div>
    """, unsafe_allow_html=True)

    # --- Title ---
    st.markdown("<h1>Emotion Recognition Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # --- Intro Section ---
    st.markdown("""
        <div class='glass'>
            <div class='intro'>
            This page allows you to test our <span style='color:#d32f2f;font-weight:600;'>Emotion Recognition Model</span> in real-time.  
            Enter any sentence or paragraph, and the system will analyze its emotional tone using our trained deep learning model.  
            It supports Recognition of <b>Joy</b>, <b>Fear</b>, <b>Anger</b>, <b>Love</b>, <b>Sadness</b>, and <b>Surprise</b>.  
            The system leverages <b>TF-IDF Vectorization</b> with a <b>Logistic Regression Classifier</b> for accurate emotion recognition.  
            </div>
        </div>
    """, unsafe_allow_html=True)

    # --- Input Box ---
    st.markdown("<div class='input-box'>", unsafe_allow_html=True)
    user_input = st.text_area("Enter your text here:", height=100, placeholder="Type something like 'I am feeling so happy today!'")
    st.markdown("</div>", unsafe_allow_html=True)

    # --- Prediction ---
    if st.button("🔍 Predict Emotion", use_container_width=True):
        if user_input.strip() == "":
            st.warning("⚠️ Please enter some text to analyze.")
        else:
            predicted_emotion, probability = predict_emotion(user_input)
            st.markdown("<div class='predict-card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='emotion'>Predicted Emotion: {predicted_emotion}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='probability'>Confidence: {probability * 100:.2f}%</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
