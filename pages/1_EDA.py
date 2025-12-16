import os
import streamlit as st
from sqlalchemy import create_engine
from src.data_models.health_news_model import HealthNewsModel
from src.controllers.health_news_controller import EDAController
from src.views.eda_view import EDAView
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="EDA - Medical Fake News",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def get_database_engine():
    """Create and cache database engine"""
    DATABASE_URL = os.getenv("DATABASE_URL", "")
    engine = create_engine(DATABASE_URL)
    return engine

@st.cache_resource
def initialize_app():
    """Initialize the MVC components"""
    engine = get_database_engine()
    model = HealthNewsModel(engine)
    controller = EDAController(model)
    view = EDAView()
    return controller, view

def main():
    controller, view = initialize_app()

    view.render_title()

    try:
        filter_options = controller.get_filter_options()
        selected_filters = view.render_filters(filter_options)

        if selected_filters is not None:
            controller.set_filters(selected_filters)
            st.cache_data.clear()

    except Exception as e:
        view.show_error(f"Error loading filter options: {str(e)}")
        st.stop()

    try:
        with view.show_loading("Loading data from database..."):
            controller.initialize_data()

        metrics = controller.get_metrics_data()
        view.render_metrics(metrics)

        df, label_col = controller.get_length_distribution_data()
        view.render_length_distribution(df, label_col)

        with view.show_loading("Generating word clouds..."):
            wordcloud_data = controller.get_wordcloud_data()
            view.render_wordclouds(wordcloud_data)

        with view.show_loading("Analyzing n-grams..."):
            df, text_col = controller.get_ngrams_data()
            view.render_ngrams(df, text_col)

        df, source_col, label_col = controller.get_source_label_data()
        view.render_source_label(df, source_col, label_col)

        with view.show_loading("Computing TF-IDF embeddings..."):
            df = controller.model.get_dataframe()
            text_col = controller.config['text_col']
            label_col = controller.config['label_col']
            source_col = controller.config['source_col']
            view.render_tfidf_embeddings(df, text_col, label_col, source_col)

    except Exception as e:
        view.show_error(f"An error occurred: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()

if __name__ == "__main__":
    main()
