import os
import streamlit as st
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="Medical Fake News Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def get_database_engine():
    """Create and cache database engine"""
    DATABASE_URL = os.getenv("DATABASE_URL", "")
    engine = create_engine(DATABASE_URL)
    return engine

# Main page content
st.title("Medical Fake News Detection System")
st.markdown("""
Welcome to the Medical Fake News Detection System. This application helps you:

- **EDA (Exploratory Data Analysis)**: Analyze and visualize the health news dataset
- **Detection**: Check if a news article is likely fake or real

Use the sidebar to navigate between pages.
""")

st.sidebar.success("Select a page above.")
