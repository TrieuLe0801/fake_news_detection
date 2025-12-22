import os
import streamlit as st
import pandas as pd
import numpy as np
import requests
from typing import Any, List, Optional, Tuple
from sqlalchemy import create_engine
from sklearn.decomposition import TruncatedSVD, PCA
from dotenv import load_dotenv

from src.processor.data_processor import clean_text, word_segmentation

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

from src.data_models.health_news_model import HealthNewsModel
from src.model_handlers.bm25_handler import BM25Handler
from src.views.detection_view import DetectionView

load_dotenv()

st.set_page_config(
    page_title="Detection - Medical Fake News",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
EMBEDDING_DIM = 256
REDUCTION_METHODS = ["SVD", "PCA", "UMAP", "t-SNE"]

# Semantic model options
SEMANTIC_MODELS = [
    "dangvantuan/vietnamese-document-embedding",
    "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base",
    "all-MiniLM-L6-v2",
    "vinai/phobert-base"
]


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

    df = model.get_all_as_dataframe()
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

    valid_df = _df[_df[text_col].notna() & (_df[text_col].str.strip() != "")].copy()

    if len(valid_df) == 0:
        return None, None

    handler.fit(valid_df, text_col)
    return handler, valid_df


@st.cache_resource
def get_dimension_reducer(
    _handler: BM25Handler,
    method: str = "SVD",
    n_components: int = EMBEDDING_DIM
) -> Optional[Any]:
    """
    Fit dimension reducer on BM25 matrix.

    Supports: SVD, PCA, UMAP, t-SNE
    Note: t-SNE is slow and not recommended for high dimensions.
    """
    if _handler is None:
        return None

    if _handler.bm25_matrix is None:
        return None

    bm25_matrix = _handler.bm25_matrix
    n_samples: int = bm25_matrix.shape[0]  # type: ignore
    n_features: int = bm25_matrix.shape[1]  # type: ignore
    actual_components = min(n_components, n_features - 1, n_samples - 1)

    if method == "SVD":
        reducer = TruncatedSVD(
            n_components=actual_components,
            random_state=42,
            algorithm="randomized"
        )
        reducer.fit(bm25_matrix)

    elif method == "PCA":
        reducer = PCA(
            n_components=actual_components,
            random_state=42
        )
        # PCA requires dense matrix
        reducer.fit(bm25_matrix.toarray())

    elif method == "UMAP":
        if not UMAP_AVAILABLE:
            st.warning("UMAP not available. Falling back to SVD.")
            return get_dimension_reducer(_handler, "SVD", n_components)

        reducer = umap.UMAP(
            n_components=actual_components,
            random_state=42,
            n_neighbors=min(15, n_samples - 1),
            min_dist=0.1,
            metric="cosine"
        )
        reducer.fit(bm25_matrix)

    elif method == "t-SNE":
        # t-SNE: First reduce with SVD, then apply t-SNE
        # Note: t-SNE doesn't have transform(), so we pre-fit SVD
        pre_reducer = TruncatedSVD(
            n_components=min(50, n_features - 1),
            random_state=42
        )
        pre_reducer.fit(bm25_matrix)

        # Store pre_reducer for transform
        reducer = {
            "type": "tsne",
            "pre_reducer": pre_reducer,
            "n_components": actual_components
        }
    else:
        raise ValueError(f"Unknown reduction method: {method}")

    return reducer


def reduce_bm25_vector(
    text: str,
    handler: BM25Handler,
    reducer: Any,
    method: str = "SVD"
) -> List[float]:
    """Transform text to reduced BM25 vector."""
    bm25_vector = handler.transform(text)

    if method == "t-SNE":
        # t-SNE doesn't support transform, use pre-reducer only
        pre_reducer = reducer["pre_reducer"]
        reduced_vector = pre_reducer.transform(bm25_vector)
    elif method == "PCA":
        # PCA requires dense
        reduced_vector = reducer.transform(bm25_vector.toarray())
    else:
        reduced_vector = reducer.transform(bm25_vector)

    return reduced_vector.flatten().tolist()


def detect_fake_news_bm25(
    text: str,
    handler: BM25Handler,
    df: pd.DataFrame,
    text_col: str = "normalized_content",
    top_k: int = 10
) -> dict:
    """Detect fake news using BM25 similarity (KNN-like approach)."""
    indices, scores = handler.query(text, top_k=top_k, return_scores=True)

    if len(indices) == 0:
        return {
            "prediction": "UNKNOWN",
            "confidence": 0.0,
            "fake_probability": 0.5,
            "similar_articles": []
        }

    similar_df = df.iloc[indices].copy()
    similar_df["similarity_score"] = scores

    total_weight = np.sum(scores)
    if total_weight > 0:
        fake_weight = np.sum(scores[similar_df["is_fake"].values == 1])
        fake_probability = fake_weight / total_weight
    else:
        fake_probability = 0.5

    prediction = "FAKE" if fake_probability >= 0.5 else "REAL"
    confidence = abs(fake_probability - 0.5) * 2

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


def detect_fake_news_hybrid(
    text: str,
    handler: BM25Handler,
    reducer: Any,
    reduction_method: str,
    api_url: str,
    semantic_model: str,
    timeout: int = 30
) -> Tuple[Optional[dict], Optional[str]]:
    """
    Detect fake news using hybrid method via API.

    Note on async: Synchronous requests are appropriate here because:
    1. Streamlit handles UI blocking with spinners
    2. Single request per detection (no parallel calls needed)
    3. Adding async would increase complexity without benefit

    Sends POST request with:
    - semantic_model_name: The semantic model to use for embedding
    - document: The input text
    - bm25_vector: BM25 vector reduced to target dimensions

    Returns:
    - result dict with label, label_text, confidence, fake_probability, real_probability
    - error message if failed
    """
    try:
        bm25_vector = reduce_bm25_vector(text, handler, reducer, reduction_method)

        payload = {
            "semantic_model_name": semantic_model,
            "document": text,
            "bm25_vector": bm25_vector
        }

        response = requests.post(
            api_url,
            json=payload,
            timeout=timeout,
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            result = response.json()
            return result, None
        else:
            error_msg = f"Status {response.status_code}: {response.text}"
            return None, error_msg

    except requests.exceptions.Timeout:
        return None, "Request timed out. Please try again."
    except requests.exceptions.ConnectionError:
        return None, "Failed to connect to API. Please check the URL."
    except Exception as e:
        return None, str(e)


def main():
    view = DetectionView()
    view.render_title()

    # Sidebar configuration
    st.sidebar.header("Detection Settings")

    detection_method = st.sidebar.radio(
        "Detection Method",
        options=["BM25 Only", "Hybrid (BM25 + Semantic)"],
        index=0,
        help="BM25 uses local similarity matching. Hybrid sends data to external API."
    )

    # Method-specific settings
    top_k = 10
    api_url = ""
    api_timeout = 30
    semantic_model = SEMANTIC_MODELS[0]
    reduction_method = "PCA"

    if detection_method == "BM25 Only":
        top_k = st.sidebar.slider(
            "Number of similar articles",
            min_value=3,
            max_value=20,
            value=10,
            help="More articles may give more stable results"
        )
    else:
        st.sidebar.subheader("API Configuration")
        api_url = st.sidebar.text_input(
            "API Endpoint URL",
            value=os.getenv("DETECTION_API_URL", "http://localhost:8000/api/detect"),
            help="URL of the hybrid detection API"
        )
        api_timeout = st.sidebar.number_input(
            "Timeout (seconds)",
            min_value=5,
            max_value=120,
            value=30
        )

        st.sidebar.subheader("Model Configuration")
        semantic_model = st.sidebar.selectbox(
            "Semantic Model",
            options=SEMANTIC_MODELS,
            index=0,
            help="Model used by API for semantic embedding"
        )

        # Dimension reduction method selector
        available_methods = REDUCTION_METHODS.copy()
        if not UMAP_AVAILABLE:
            available_methods.remove("UMAP")

        reduction_method = st.sidebar.selectbox(
            "Dimension Reduction Method",
            options=available_methods,
            index=1,
            help="""
            - SVD: Fast, good for sparse matrices (Recommended)
            - PCA: Standard, requires dense conversion
            - UMAP: Preserves local structure, slower
            - t-SNE: Good visualization, no transform support
            """
        )

    text_col = st.sidebar.selectbox(
        "Text column",
        options=["normalized_content", "content"],
        index=0,
        help="normalized_content is recommended for Vietnamese text"
    )

    # Load data and handlers
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

        # Initialize reducer for hybrid method
        reducer = None
        if detection_method == "Hybrid (BM25 + Semantic)":
            with st.spinner(f"Initializing {reduction_method} reducer ({EMBEDDING_DIM} dimensions)..."):
                reducer = get_dimension_reducer(handler, reduction_method, EMBEDDING_DIM)

            if reducer is None:
                st.error(f"Failed to initialize {reduction_method} reducer.")
                return

            st.sidebar.divider()
            st.sidebar.subheader("Embedding Info")
            if hasattr(handler, 'bm25_matrix') and handler.bm25_matrix is not None:
                st.sidebar.metric("BM25 Features", handler.bm25_matrix.shape[1])  # type: ignore

            # Get actual dimensions
            if reduction_method == "t-SNE":
                actual_dims = reducer["pre_reducer"].n_components_
            else:
                actual_dims = getattr(reducer, 'n_components_', getattr(reducer, 'n_components', EMBEDDING_DIM))
            st.sidebar.metric("Reduced Dimensions", actual_dims)
            st.sidebar.metric("Reduction Method", reduction_method)

    except Exception as e:
        view.render_error(f"Failed to load data: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return

    # Render input form
    submitted_text = view.render_input_form()

    if submitted_text:
        submitted_text = word_segmentation(
                clean_text(submitted_text), lib="pyvi", remove_stopwords=True
            )
        with view.render_loading():
            try:
                if detection_method == "BM25 Only":
                    result = detect_fake_news_bm25(
                        text=submitted_text,
                        handler=handler,
                        df=valid_df,  # type: ignore
                        text_col=text_col,
                        top_k=top_k
                    )
                    view.render_result(result)
                else:
                    result, error = detect_fake_news_hybrid(
                        text=submitted_text,
                        handler=handler,
                        reducer=reducer,
                        reduction_method=reduction_method,
                        api_url=api_url,
                        semantic_model=semantic_model,
                        timeout=int(api_timeout)
                    )

                    if error:
                        view.render_api_error(error)
                    elif result:
                        view.render_hybrid_result(result)

            except Exception as e:
                view.render_error(f"Detection failed: {str(e)}")
                import traceback
                st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
