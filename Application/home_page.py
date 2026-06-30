import streamlit as st
from pathlib import Path
from PIL import Image

def show_home_page():
    # --- Page Config ---
    st.set_page_config(page_title="Emotion Recognition System", page_icon="🧠", layout="wide")

    # --- Resolve Image Paths ---
    base_path = Path(__file__).parent / "assets"
    supervisor_img = base_path / "supervisor.jpg"
    member1_img = base_path / "member01.jpg"
    member2_img = base_path / "member02.jpg"
    tech_imgs = {
        "Python": base_path / "python.png",
        "NumPy": base_path / "numpy.png",
        "Scikit-Learn": base_path / "scikit-learn.png",
        "Pandas": base_path / "pandas.png",
        "Streamlit": base_path / "streamlit.png",
    }

    # --- Custom CSS ---
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

        .subtitle {
            text-align: center;
            color: #555;
            font-size: 20px;
            margin-bottom: 30px;
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
        .highlight {
            color: #d32f2f;
            font-weight: 700;
        }
        .divider {
            height: 3px;
            width: 200px;
            background: linear-gradient(to right, #e53935, #ef5350);
            border-radius: 5px;
            margin: 15px auto 30px;
        }
        .footer {
            text-align: center;
            margin-top: 60px;
            font-size: 15px;
            color: #444;
            background: rgba(255,255,255,0.7);
            border-radius: 12px;
            padding: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
            width: 70%;
            margin-left: auto;
            margin-right: auto;
        }
        .center-image {
            display: flex;
            justify-content: center;
        }
        </style>
    """, unsafe_allow_html=True)

    # --- Header Bar ---
    st.markdown("""
        <div class='header-bar'>
            <div class='header-title'>🎓 Emotion Recognition Using Deep Learning</div>
            <div class='header-subtitle'>Final Year Project • Department of Computer Science</div>
        </div>
    """, unsafe_allow_html=True)

    # --- Title ---
    st.markdown("<h1>Emotion Recognition Using Deep Learning</h1>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Experience AI that understands your emotions ✨</div>", unsafe_allow_html=True)
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # --- Intro Section ---
    st.markdown("""
        <div class='glass'>
            <div class='intro'>
            Welcome to the <span class='highlight'>Emotion Recognition System</span> —  
            an advanced AI-powered platform that analyzes human emotions through text using deep learning techniques.  
            This intelligent system identifies key emotional states such as <b>happiness, anger, sadness, fear, surprise,</b> and <b>disgust</b> from textual input,  
            providing insights into the writer’s emotional tone.  
            <br><br>
            The model has been trained on large emotion-labeled datasets using neural networks that capture semantic meaning  
            and contextual relationships between words.  
            It serves as a bridge between human communication and machine understanding — enabling smarter interaction in  
            applications such as chatbots, mental health monitoring, and customer feedback systems.  
            <br><br>
            Developed with <b>Python</b> and deployed using <b>Streamlit</b>, the project integrates advanced preprocessing,  
            vectorization, and predictive modeling for robust and real-time emotion recognition.
            </div>
        </div>
    """, unsafe_allow_html=True)

    # --- Function for fixed image size ---
    def fixed_image(image_path, caption, size=(250, 250)):
        if image_path.exists():
            img = Image.open(image_path).resize(size)
            st.markdown("<div class='center-image'>", unsafe_allow_html=True)
            st.image(img, caption=caption, use_container_width=False)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning(f"Image not found: {image_path.name}")

    # --- Supervisor Section ---
    st.markdown("<h2>Supervisor</h2>", unsafe_allow_html=True)
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    fixed_image(supervisor_img, "Mr. Umair Ali")
    st.markdown("""
        <div class='intro' style='margin-top:15px; text-align:center; width:70%; margin:auto;'>
        I am honoured to have worked under the supervision of <b>Mr. Umair Ali</b>,  
       
        </div>
    """, unsafe_allow_html=True)

    # --- Group Members Section ---
    st.markdown("<h2>Group Members</h2>", unsafe_allow_html=True)
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    fixed_image(member1_img, "Anila Batool")
    st.markdown("""
        <div class='intro' style='width:70%; margin:auto; text-align:center;'>
        I am a dedicated learner with a strong passion for Artificial Intelligence.  
        My focus in this project was on model training, data preprocessing, and emotion recognition using deep learning models.
        </div>
    """, unsafe_allow_html=True)

    
    # --- Technologies Section ---
    st.markdown("<h2>Languages & Libraries Used</h2>", unsafe_allow_html=True)
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='center-image'>", unsafe_allow_html=True)
    tech_cols = st.columns(5)
    for i, (name, img_path) in enumerate(tech_imgs.items()):
        with tech_cols[i]:
            if img_path.exists():
                st.image(str(img_path), caption=name, width=90, use_container_width=False)
            else:
                st.markdown(f"**{name}**<br><small>(image missing)</small>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # --- Footer ---
    st.markdown("""
        <div class='footer'>
            Developed with ❤️ by <b>Anila Batool</b></b><br>
             Built with <b>Streamlit</b> & <b>Deep Learning</b>
        </div>
    """, unsafe_allow_html=True)
