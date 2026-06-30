import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.stem import PorterStemmer
import re

# Download stopwords (only once)
nltk.download('stopwords', quiet=True)
stopwords = set(nltk.corpus.stopwords.words('english'))

# ========================== Load Data ================================
@st.cache_data
def load_data():
    df = pd.read_csv("train.txt", header=None, sep=";", names=["Comment", "Emotion"], encoding="utf-8")
    df['length'] = df['Comment'].apply(len)
    return df

df = load_data()

# ========================== Text Cleaning =============================
def clean_text(text):
    stemmer = PorterStemmer()
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower().split()
    text = [stemmer.stem(word) for word in text if word not in stopwords]
    return " ".join(text)


# ========================== Main Explore Page =============================
def show_explore_page():
    # --- Page Config ---
    st.set_page_config(page_title="Explore Dataset", page_icon="📊", layout="wide")

    # --- Red Header Styling (same as home) ---
    st.markdown("""
        <style>
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
        </style>
    """, unsafe_allow_html=True)

    # --- Red Header Bar (same as home) ---
    st.markdown("""
        <div class='header-bar'>
            <div class='header-title'>📊 Explore Emotion Recognition Dataset</div>
            <div class='header-subtitle'>Visual Insights into Text Data and Emotional Patterns</div>
        </div>
    """, unsafe_allow_html=True)

    # ================== Main Content (unchanged) ==================
    st.title("📊 Explore Emotion Recognition Dataset")
    st.write("Gain insights into the dataset used for training the Emotion Recognition model.")
    st.markdown("---")

    # ========== Dataset Preview ==========
    st.subheader("1️⃣ Dataset Preview")
    st.dataframe(df.head(10))

    st.markdown("---")

    # ========== Emotion Distribution ==========
    st.subheader("2️⃣ Emotion Distribution (Bar Chart)")
    emotion_counts = df["Emotion"].value_counts()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=emotion_counts.index, y=emotion_counts.values, palette="coolwarm", ax=ax)
    ax.set_xlabel("Emotion", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution of Emotions", fontsize=14)
    for i, v in enumerate(emotion_counts.values):
        ax.text(i, v + 20, str(v), ha='center', fontsize=10)
    st.pyplot(fig)

    st.markdown("### 🥧 Emotion Distribution (Pie Chart)")
    fig, ax = plt.subplots()
    ax.pie(
        emotion_counts,
        labels=emotion_counts.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=sns.color_palette("coolwarm", len(emotion_counts))
    )
    ax.axis("equal")
    st.pyplot(fig)

    st.markdown("---")

    # ========== Comment Length Distribution ==========
    st.subheader("3️⃣ Comment Length Distribution by Emotion")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data=df, x="length", hue="Emotion", multiple="stack", bins=50, palette="Spectral", ax=ax)
    ax.set_title("Comment Length Distribution by Emotion", fontsize=14)
    ax.set_xlabel("Comment Length")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    st.markdown("---")

    # ========== Boxplot ==========
    st.subheader("4️⃣ Comment Length Boxplot by Emotion")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x="Emotion", y="length", palette="viridis", ax=ax)
    ax.set_title("Comment Length Spread for Each Emotion", fontsize=14)
    ax.set_xlabel("Emotion")
    ax.set_ylabel("Comment Length")
    st.pyplot(fig)

    st.markdown("---")

    # ========== Word Clouds ==========
    st.subheader("5️⃣ Word Cloud for Each Emotion")
    st.write("Visual representation of the most frequent words per emotion:")
    emotions = df['Emotion'].unique()

    for emotion in emotions:
        text = " ".join(df.loc[df['Emotion'] == emotion, 'Comment'])
        text = clean_text(text)
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='coolwarm',
            max_words=100
        ).generate(text)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_title(f"{emotion} Word Cloud", fontsize=16)
        ax.axis("off")
        st.pyplot(fig)
