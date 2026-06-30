import streamlit as st
import pickle
import matplotlib.pyplot as plt

def show_performance_page():
    # --- Page Config ---
    st.set_page_config(page_title="Model Performance", page_icon="📈", layout="wide")

    # --- Custom Styling ---
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
        /* --- Red Header Bar --- */
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

        /* --- Headings --- */
        h1, h2 {
            text-align: center;
            color: #b71c1c;
            font-weight: 700;
        }
        h1 {
            font-size: 52px;
            margin-bottom: 0px;
        }
        h2 {
            font-size: 32px;
            margin-top: 50px;
        }

        .divider {
            height: 3px;
            width: 200px;
            background: linear-gradient(to right, #e53935, #ef5350);
            border-radius: 5px;
            margin: 15px auto 30px;
        }

        /* --- Intro Card (like Home Page) --- */
        .intro-card {
            background-color: #f9f9f9;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
            text-align: justify;
            font-size: 18px;
            line-height: 1.8;
            width: 75%;
            margin: auto;
            color: #222;
        }

        .footer {
            text-align: center;
            margin-top: 40px;
            font-size: 15px;
            color: #555;
            background: rgba(255,255,255,0.7);
            border-radius: 12px;
            padding: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
            width: 70%;
            margin-left: auto;
            margin-right: auto;
        }
        </style>
    """, unsafe_allow_html=True)

    # --- Red Header Bar ---
    st.markdown("""
        <div class='header-bar'>
            <div class='header-title'>📊 Model Training Performance</div>
            <div class='header-subtitle'>Accuracy and Loss Trends Across Epochs</div>
        </div>
    """, unsafe_allow_html=True)

    # --- Page Title and Intro Section (with card) ---
    st.markdown("<h1>Model Learning & Evaluation</h1>", unsafe_allow_html=True)
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("""
        <div class='intro-card'>
        The <b>Emotion Recognition Model</b> has been trained using deep learning techniques to analyze emotional tone from text.  
        During model training, it learns to associate linguistic features and contextual patterns with emotion categories.  
        This page visualizes how the model’s accuracy and loss evolved through training and validation stages —  
        helping us understand how effectively it generalized to unseen data.  
        </div>
    """, unsafe_allow_html=True)

    # --- Load Training Metrics ---
    try:
        with open("training_metrics.pkl", "rb") as f:
            metrics = pickle.load(f)
    except FileNotFoundError:
        st.error("❌ 'training_metrics.pkl' file not found. Please train your model and save metrics first.")
        return

    # --- Accuracy Graph ---
    if "train_accuracy" in metrics and "val_accuracy" in metrics:
        st.markdown("<h2>🔹 Training vs Validation Accuracy</h2>", unsafe_allow_html=True)
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        fig1, ax1 = plt.subplots(figsize=(6, 3))
        ax1.plot(metrics["train_accuracy"], label="Training Accuracy", linewidth=2, marker='o', color='#0077b6')
        ax1.plot(metrics["val_accuracy"], label="Validation Accuracy", linewidth=2, marker='s', color='#00b4d8')
        ax1.set_title("Training & Validation Accuracy per Epoch", fontsize=14, color='#b71c1c')
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Accuracy")
        ax1.grid(alpha=0.3)
        ax1.legend()
        st.pyplot(fig1)
        st.markdown("""
            <div class='intro' style='text-align:center; width:70%; margin:auto;'>
            The accuracy graph illustrates how the model improved its emotional understanding with each epoch.  
            Stable convergence between training and validation accuracy reflects strong learning and reduced overfitting.
            </div>
        """, unsafe_allow_html=True)

    # --- Loss Graph ---
    if "train_loss" in metrics and "val_loss" in metrics:
        st.markdown("<h2>🔹 Training vs Validation Loss</h2>", unsafe_allow_html=True)
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        ax2.plot(metrics["train_loss"], label="Training Loss", linewidth=2, marker='o', color='#ff6f00')
        ax2.plot(metrics["val_loss"], label="Validation Loss", linewidth=2, marker='s', color='#e63946')
        ax2.set_title("Training & Validation Loss per Epoch", fontsize=14, color='#b71c1c')
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Loss")
        ax2.grid(alpha=0.3)
        ax2.legend()
        st.pyplot(fig2)
        st.markdown("""
            <div class='intro' style='text-align:center; width:70%; margin:auto;'>
            The loss curve shows how the model minimized prediction errors across epochs.  
            When training and validation loss stabilize together, it signifies optimal generalization and steady learning.
            </div>
        """, unsafe_allow_html=True)

    # --- Footer ---
    st.markdown("""
        <div class='footer'>
            🚀 Visualizing the learning journey of our deep learning model.  
    
        </div>
    """, unsafe_allow_html=True)
