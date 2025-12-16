import os
import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from dotenv import load_dotenv

from src.data_models.health_news_model import HealthNewsModel
from src.model_handlers.bm25_handler import BM25Handler
from src.views.detection_view import DetectionView

load_dotenv()

st.set_page_config(
    page_title="Detection - Medical Fake News",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def get_database_engine():
    """Create and cache database engine."""
    DATABASE_URL = os.getenv("DATABASE_URL", "")
    engine = create_engine(DATABASE_URL)
    return engine


@st.cache_resource
def load_labeled_data():
    """Load labeled data from database."""
    engine = get_database_engine()
    model = HealthNewsModel(engine)

    # Load all data
    df = model.get_all_as_dataframe()

    # Filter to only labeled data (is_fake is not null)
    labeled_df = df[df["is_fake"].notna()].copy()
    labeled_df["is_fake"] = labeled_df["is_fake"].astype(int)

    return labeled_df


@st.cache_resource
def get_bm25_handler(_df: pd.DataFrame, text_col: str = "normalized_content"):
    """Fit BM25 handler on labeled data."""
    handler = BM25Handler(
        max_features=5000,
        method="lucene",
        k1=1.5,
        b=0.75,
        min_df=2,
        max_df=0.95,
        n_jobs=-1
    )

    # Filter out rows with empty text
    valid_df = _df[_df[text_col].notna() & (_df[text_col].str.strip() != "")].copy()

    if len(valid_df) == 0:
        return None, None

    handler.fit(valid_df, text_col)
    return handler, valid_df


def detect_fake_news(
    text: str,
    handler: BM25Handler,
    df: pd.DataFrame,
    text_col: str = "normalized_content",
    top_k: int = 10
) -> dict:
    """
    Detect if news is fake using BM25 similarity.

    Uses a KNN-like approach:
    1. Find top-K most similar articles using BM25
    2. Weight votes by similarity score
    3. Calculate fake probability based on weighted voting
    """
    # Query similar documents
    indices, scores = handler.query(text, top_k=top_k, return_scores=True)

    if len(indices) == 0:
        return {
            "prediction": "UNKNOWN",
            "confidence": 0.0,
            "fake_probability": 0.5,
            "similar_articles": []
        }

    # Get labels of similar documents
    similar_df = df.iloc[indices].copy()
    similar_df["similarity_score"] = scores

    # Calculate weighted fake probability
    total_weight = np.sum(scores)
    if total_weight > 0:
        fake_weight = np.sum(scores[similar_df["is_fake"].values == 1])
        fake_probability = fake_weight / total_weight
    else:
        fake_probability = 0.5

    # Make prediction
    prediction = "FAKE" if fake_probability >= 0.5 else "REAL"

    # Calculate confidence (how far from 0.5)
    confidence = abs(fake_probability - 0.5) * 2

    # Prepare similar articles for display
    similar_articles = []
    for rank, (idx, row) in enumerate(similar_df.iterrows(), 1):
        similar_articles.append({
            "rank": rank,
            "title": row.get("title", "N/A")[:100],
            "source": row.get("source", "Unknown"),
            "label": int(row["is_fake"]),
            "similarity_score": row["similarity_score"]
        })

    return {
        "prediction": prediction,
        "confidence": confidence,
        "fake_probability": fake_probability,
        "similar_articles": similar_articles
    }


def main():
    view = DetectionView()
    view.render_title()

    # Sidebar configuration
    st.sidebar.header("Detection Settings")
    top_k = st.sidebar.slider(
        "Number of similar articles to consider",
        min_value=3,
        max_value=20,
        value=10,
        help="More articles may give more stable results but slower"
    )

    text_col = st.sidebar.selectbox(
        "Text column for similarity",
        options=["normalized_content", "content"],
        index=0,
        help="normalized_content is recommended for Vietnamese text"
    )

    # Load data and BM25 handler
    try:
        with st.spinner("Loading labeled data..."):
            labeled_df = load_labeled_data()

        if len(labeled_df) == 0:
            view.render_no_data_warning()
            return

        # Show data statistics
        st.sidebar.divider()
        st.sidebar.subheader("Database Statistics")
        st.sidebar.metric("Total Labeled Articles", len(labeled_df))
        fake_count = (labeled_df["is_fake"] == 1).sum()
        real_count = (labeled_df["is_fake"] == 0).sum()
        st.sidebar.metric("Fake Articles", fake_count)
        st.sidebar.metric("Real Articles", real_count)

        with st.spinner("Initializing BM25 model..."):
            handler, valid_df = get_bm25_handler(labeled_df, text_col)

        if handler is None:
            st.error("Failed to initialize BM25 handler. No valid text data found.")
            return

    except Exception as e:
        view.render_error(f"Failed to load data: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return

    # Render input form
    submitted_text = view.render_input_form()

    if submitted_text:
        with view.render_loading():
            try:
                result = detect_fake_news(
                    text=submitted_text,
                    handler=handler,
                    df=valid_df,
                    text_col=text_col,
                    top_k=top_k
                )
                view.render_result(result)
            except Exception as e:
                view.render_error(f"Detection failed: {str(e)}")
                import traceback
                st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
