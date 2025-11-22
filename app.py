import os
import streamlit as st
from sqlalchemy import create_engine
from src.data_models.health_news_model import HealthNewsModel
from src.controllers.health_news_controller import EDAController
from src.views.eda_view import EDAView
from dotenv import load_dotenv

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Medical Fake News EDA",
    # page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def get_database_engine():
    """Create and cache database engine"""
    # Replace with your actual database connection string
    DATABASE_URL = os.getenv("DATABASE_URL", "")
    engine = create_engine(DATABASE_URL)
    return engine

@st.cache_resource
def initialize_app():
    """Initialize the MVC components"""
    # Get database engine
    engine = get_database_engine()
    
    # Initialize model (combines repository + service)
    model = HealthNewsModel(engine)
    
    # Initialize controller
    controller = EDAController(model)
    
    # Initialize view with plotting functions
    view = EDAView()
    
    return controller, view

def main():
    # Initialize MVC components
    controller, view = initialize_app()
    
    # Render title
    view.render_title()
    
    # Get filter options
    try:
        filter_options = controller.get_filter_options()
        
        # Render filters and get selected values
        selected_filters = view.render_filters(filter_options)
        
        # Apply filters if changed
        if selected_filters is not None:
            controller.set_filters(selected_filters)
            st.cache_data.clear()  # Clear cache when filters change
        
    except Exception as e:
        view.show_error(f"Error loading filter options: {str(e)}")
        st.stop()
    
    # Load and display data
    try:
        with view.show_loading("Loading data from database..."):
            controller.initialize_data()
        
        # Render metrics
        metrics = controller.get_metrics_data()
        view.render_metrics(metrics)
        
        # Render length distribution
        df, label_col = controller.get_length_distribution_data()
        view.render_length_distribution(df, label_col)
        
        # Render word clouds
        with view.show_loading("Generating word clouds..."):
            wordcloud_data = controller.get_wordcloud_data()
            view.render_wordclouds(wordcloud_data)
        
        # Render n-grams
        with view.show_loading("Analyzing n-grams..."):
            df, text_col = controller.get_ngrams_data()
            view.render_ngrams(df, text_col)
        
        # Render source-label distribution
        df, source_col, label_col = controller.get_source_label_data()
        view.render_source_label(df, source_col, label_col)
        
    except Exception as e:
        view.show_error(f"An error occurred: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()

if __name__ == "__main__":
    main()