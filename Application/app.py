import streamlit as st
from predict_page import show_predict_page
from explore_page import show_explore_page
from home_page import show_home_page
from evaluation_page import show_performance_page

# --- Sidebar Navigation ---
st.sidebar.title("🧭 Navigation")
page = st.sidebar.selectbox(
    "Select a Page",
    ("Home", "Predict", "Explore", "Training")
)

# --- Page Routing ---
if page == "Home":
    show_home_page()
elif page == "Predict":
    show_predict_page()
elif page == "Explore":
    show_explore_page()
elif page == "Training":
    show_performance_page()
