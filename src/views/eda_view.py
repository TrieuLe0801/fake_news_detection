from datetime import date, datetime
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt

from src.views.data_visualizer import (
    compute_embeddings,
    generate_wordcloud,
    ngram_by_label,
    plot_embedding_3d,
    plot_embedding_3d_with_sources,
    plot_embedding_scatter,
    plot_embedding_scatter_with_sources,
    plot_length_distribution,
    plot_source_label_counts,
    plot_source_label_counts_grouped,
    plot_top_ngrams,
    plot_top_weighted_terms,
    reduce_dimensions,
    wordcloud_by_label,
)


class EDAView:
    """View layer - handles UI rendering"""

    def __init__(self):
        self.viz = {
            "length_distribution": plot_length_distribution,
            "wordcloud": generate_wordcloud,
            "top_ngrams": plot_top_ngrams,
            "ngrams_label": ngram_by_label,
            "source_label": plot_source_label_counts_grouped,
            "reduce_dimensions": reduce_dimensions,
            "compute_embeddings": compute_embeddings,
            "plot_embedding_scatter_with_sources": plot_embedding_scatter_with_sources,
            "plot_embedding_scatter": plot_embedding_scatter,
            "plot_embedding_3d_with_sources": plot_embedding_3d_with_sources,
            "plot_embedding_3d": plot_embedding_3d,
            "plot_top_weighted_terms": plot_top_weighted_terms,
        }

    def render_title(self):
        """Render page title"""
        st.title("Medical Fake News EDA Dashboard")
        st.divider()

    def render_filters(self, filter_options: Dict[str, Any]) -> Dict[str, Any]:
        """Render filter sidebar and return selected filters"""
        with st.sidebar:
            st.header("Filters")

            # Date range filter
            if filter_options["min_date"] and filter_options["max_date"]:
                date_range = st.date_input(
                    "Date Range",
                    value=(filter_options["min_date"], filter_options["max_date"]),
                    min_value=filter_options["min_date"],
                    max_value=filter_options["max_date"],
                )
            else:
                date_range = None

            # Source filter
            sources = st.multiselect("Sources", options=filter_options["sources"], default=None)

            # Label filter
            label_filter = st.selectbox(
                "News Type", options=["Real News", "Fake News", "All"], index=2
            )

            apply_filters = st.button("Apply Filters", use_container_width=True)

            filters = {}
            if date_range and len(date_range) == 2:
                filters["start_date"] = datetime.combine(date_range[0], datetime.min.time())
                filters["end_date"] = datetime.combine(date_range[1], datetime.max.time())

            if sources:
                filters["sources"] = sources

            if label_filter == "Real News":
                filters["is_fake"] = False
            elif label_filter == "Fake News":
                filters["is_fake"] = True

            return filters if apply_filters else None

    def render_metrics(self, metrics: Dict[str, int]):
        """Render summary metrics"""
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Articles", f"{metrics['total_articles']:,}")
        with col2:
            st.metric("Fake News", f"{metrics['fake_news']:,}")
        with col3:
            st.metric("Real News", f"{metrics['real_news']:,}")
        with col4:
            st.metric("Sources", metrics["num_sources"])

        st.divider()

    def render_length_distribution(self, df, label_col):
        """Render length distribution section"""
        st.header("Text Length Distribution")
        fig = self.viz["length_distribution"](df, label_col)
        st.pyplot(fig)
        plt.close(fig)  # Clean up
        st.divider()

    def render_wordclouds(self, wordcloud_data: Dict[str, Any]):
        """Render word cloud section"""
        st.header("Word Clouds")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("All Content")
            fig = self.viz["wordcloud"](
                wordcloud_data["all"], title="All Articles", stopwords=set()
            )
            st.pyplot(fig)
            plt.close(fig)

        labels = list(wordcloud_data["by_label"].keys())
        label_names = {0: "Real News", 1: "Fake News"}

        with col2:
            if len(labels) > 0:
                label = labels[0]
                st.subheader(label_names.get(label, f"Label {label}"))
                fig = self.viz["wordcloud"](
                    wordcloud_data["by_label"][label],
                    title=label_names.get(label, f"Label {label}"),
                    stopwords=set(),
                )
                st.pyplot(fig)
                plt.close(fig)

        with col3:
            if len(labels) > 1:
                label = labels[1]
                st.subheader(label_names.get(label, f"Label {label}"))
                fig = self.viz["wordcloud"](
                    wordcloud_data["by_label"][label],
                    title=label_names.get(label, f"Label {label}"),
                    stopwords=set(),
                )
                st.pyplot(fig)
                plt.close(fig)

        st.divider()

    def render_ngrams(self, df, text_col):
        """Render n-grams section"""
        st.header("N-grams Analysis")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Top 30 Unigrams")
            fig = self.viz["top_ngrams"](df, text_col, ngram_range=(1, 1), top_n=30)
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            st.subheader("Top 30 Bigrams")
            fig = self.viz["top_ngrams"](df, text_col, ngram_range=(2, 2), top_n=30)
            st.pyplot(fig)
            plt.close(fig)

        st.divider()

    def render_ngrams_by_label(self, df, text_col, label_col):
        """Render n-grams by label section (optional)"""
        st.header("N-grams Analysis by Label")

        # Unigrams by label
        st.subheader("Unigrams by Label")
        unigram_figs = self.viz["ngram_label"](
            df, text_col, label_col, ngram_range=(1, 1), top_n=15
        )

        cols = st.columns(len(unigram_figs))
        for idx, (label, fig) in enumerate(unigram_figs.items()):
            with cols[idx]:
                label_name = (
                    "Real News" if label == 0 else "Fake News" if label == 1 else f"Label {label}"
                )
                st.write(f"**{label_name}**")
                st.pyplot(fig)
                plt.close(fig)

        st.divider()

        # Bigrams by label
        st.subheader("Bigrams by Label")
        bigram_figs = self.viz["ngrams_label"](
            df, text_col, label_col, ngram_range=(2, 2), top_n=15
        )

        cols = st.columns(len(bigram_figs))
        for idx, (label, fig) in enumerate(bigram_figs.items()):
            with cols[idx]:
                label_name = (
                    "Real News" if label == 0 else "Fake News" if label == 1 else f"Label {label}"
                )
                st.write(f"**{label_name}**")
                st.pyplot(fig)
                plt.close(fig)

        st.divider()

    def render_source_label(self, df, source_col, label_col):
        """Render source-label analysis section"""
        st.header("Source vs Label Distribution")
        fig = self.viz["source_label"](df, source_col, label_col)
        st.pyplot(fig)
        plt.close(fig)
        st.divider()

    def render_tfidf_embeddings(self, df, text_col, label_col, source_col):
        """Render TF-IDF/BM25 embeddings visualization section"""
        st.header("Text Embeddings Visualization")

        # Initialize session state for caching
        if "embedding_cache" not in st.session_state:
            st.session_state.embedding_cache = {}

        # Settings in sidebar
        with st.sidebar:
            st.subheader("Embedding Settings")

            # Embedding Method Selection
            embedding_method = st.radio(
                "Embedding Method",
                options=["TF-IDF", "BM25"],
                index=1,
                help="""
                **TF-IDF:** Classic method, linear weighting
                **BM25:** Modern method, non-linear saturation, better for ranking
                """,
            )

            reduction_method = st.selectbox(
                "Reduction Method", options=["SVD", "PCA", "UMAP", "t-SNE"], index=1
            )

            n_components = st.radio(
                "Dimensions",
                options=[2, 3],
                index=1,
                help="2D for simple visualization, 3D for more detail",
            )

            # max_features = st.slider(
            #     "Max TF-IDF Features",
            #     min_value=1000,
            #     max_value=10000,
            #     value=5000,
            #     step=1000,
            #     help="Maximum number of TF-IDF features",
            # )
            max_features = 5000  # Fixed to 5000 for BM25 optimization

            show_sources = st.checkbox("Color by Source", value=False)
            show_top_terms = st.checkbox("Show Top Weighted Terms", value=True)

            # 3D specific settings
        if n_components == 3:
            st.divider()
            st.subheader("3D Settings")
            rotation_angle = st.slider(
                "Rotation Angle",
                min_value=0,
                max_value=360,
                value=45,
                step=15,
                help="Rotate the 3D plot",
            )
            elevation_angle = st.slider(
                "Elevation Angle",
                min_value=-90,
                max_value=90,
                value=20,
                step=10,
                help="Viewing elevation",
            )

        # Cache management
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("Recompute", help="Force recompute embeddings"):
                st.session_state.embedding_cache = {}
                st.rerun()
        with col2:
            cache_size = len(st.session_state.embedding_cache)
            st.metric("Cached", cache_size)

        # Create cache key based on settings that affect computation
        cache_key = (
            embedding_method,
            max_features,
            reduction_method,
            n_components,
            len(df),  # Include data size
            hash(df[text_col].iloc[0]) if len(df) > 0 else 0,  # Sample hash
        )

        # Check if we need to recompute
        need_recompute = cache_key not in st.session_state.embedding_cache

        if need_recompute:
            # Show what's being computed
            st.info(
                f"""
            Computing embeddings:
            - Method: **{embedding_method}**
            - Features: **{max_features}**
            - Reduction: **{reduction_method}**
            - Dimensions: **{n_components}D**
            """
            )

            # Get latest cache
            if len(st.session_state.embedding_cache) > 0:
                cache_list = sorted(
                    st.session_state.embedding_cache.items(),
                    key=lambda x: x[1]["timestamp"],
                    reverse=True,
                )
                latest_cache = cache_list[0][1] if cache_list else None
                print(latest_cache)

                # Compute embeddings
                if (
                    latest_cache
                    and latest_cache["embedding_method"] == embedding_method
                    and latest_cache["max_features"] == max_features
                ):
                    # Reuse matrix if embedding method and features match
                    matrix = latest_cache["matrix"]
                    vectorizer = latest_cache["vectorizer"]
                else:
                    with st.spinner(f"Computing {embedding_method} embeddings..."):
                        # Step 1: Compute TF-IDF/BM25 matrix
                        matrix, vectorizer = self.viz["compute_embeddings"](
                            df,
                            text_col,
                            method=embedding_method.lower().replace("-", ""),
                            max_features=max_features,
                        )
                if (
                    latest_cache
                    and latest_cache["reduction_method"] == reduction_method
                    and latest_cache["dimensions"] == n_components
                ):
                    # Reuse embeddings if reduction method and dimensions match
                    embeddings = latest_cache["embeddings"]
                else:
                    with st.spinner(f"Reducing dimensions with {reduction_method}..."):
                        # Step 2: Reduce dimensions
                        embeddings = self.viz["reduce_dimensions"](
                            matrix,
                            method=reduction_method.lower().replace("-", ""),
                            n_components=n_components,
                        )
            else:
                with st.spinner(f"Computing {embedding_method} embeddings..."):
                    # Step 1: Compute TF-IDF/BM25 matrix
                    matrix, vectorizer = self.viz["compute_embeddings"](
                        df,
                        text_col,
                        method=embedding_method.lower().replace("-", ""),
                        max_features=max_features,
                    )
                with st.spinner(f"Reducing dimensions with {reduction_method}..."):
                    # Step 2: Reduce dimensions
                    embeddings = self.viz["reduce_dimensions"](
                        matrix,
                        method=reduction_method.lower().replace("-", ""),
                        n_components=n_components,
                    )

            # Cache the results
            st.session_state.embedding_cache[cache_key] = {
                "embedding_method": embedding_method,
                "max_features": max_features,
                "reduction_method": reduction_method,
                "dimensions": n_components,
                "matrix": matrix,
                "vectorizer": vectorizer,
                "embeddings": embeddings,
                "timestamp": pd.Timestamp.now(),
            }

            st.success("Embeddings computed and cached!")

        else:
            # Use cached results
            cached_data = st.session_state.embedding_cache[cache_key]
            matrix = cached_data["matrix"]
            vectorizer = cached_data["vectorizer"]
            embeddings = cached_data["embeddings"]

            st.success(
                f"Using cached embeddings (computed at {cached_data['timestamp'].strftime('%H:%M:%S')})"
            )

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Documents", len(df))
        with col2:
            st.metric("Features", matrix.shape[1])
        with col3:
            st.metric("Embedding Method", embedding_method)
        with col4:
            sparsity = 1.0 - (matrix.nnz / (matrix.shape[0] * matrix.shape[1]))
            st.metric("Sparsity", f"{sparsity*100:.1f}%")
        st.divider()

        # Plot embeddings
        if n_components == 2:
            # 2D plots
            if show_sources:
                st.subheader(f"2D {reduction_method} Embeddings - By Source and Label")
                fig = self.viz["plot_embedding_scatter_with_sources"](
                    df,
                    embeddings,
                    source_col,
                    label_col,
                    reduction_method=reduction_method,
                    embedding_method=embedding_method,
                )
            else:
                st.subheader(f"2D {reduction_method} Embeddings - By Label")
                fig = self.viz["plot_embedding_scatter"](
                    df,
                    embeddings,
                    label_col,
                    reduction_method=reduction_method,
                    embedding_method=embedding_method,
                )
            st.pyplot(fig)
            plt.close(fig)

        else:  # 3D plots
            if show_sources:
                st.subheader(f"3D {reduction_method} Embeddings - By Source and Label")

                # Create plot with custom viewing angles
                fig = self.viz["plot_embedding_3d_with_sources"](
                    df,
                    embeddings,
                    source_col,
                    label_col,
                    reduction_method=reduction_method,
                    embedding_method=embedding_method,
                )

                # Update viewing angle if user changed it
                if "rotation_angle" in locals() and "elevation_angle" in locals():
                    ax = fig.gca()
                    ax.view_init(elev=elevation_angle, azim=rotation_angle)

            else:
                st.subheader(f"3D {reduction_method} Embeddings - By Label")

                # Create plot
                fig = self.viz["plot_embedding_3d"](
                    df,
                    embeddings,
                    label_col,
                    reduction_method=reduction_method,
                    embedding_method=embedding_method,
                )

                # Update viewing angle if user changed it
                if "rotation_angle" in locals() and "elevation_angle" in locals():
                    ax = fig.gca()
                    ax.view_init(elev=elevation_angle, azim=rotation_angle)

            st.pyplot(fig)
            plt.close(fig)

            # Add instructions for 3D
            st.info(
                """
            **3D Viewing Tips:**
            - Use the sliders in the sidebar to rotate the plot
            - Look for cluster separation in 3D space
            - 3D can reveal patterns not visible in 2D
            """
            )

        # Show top weighted terms
        if show_top_terms:
            st.markdown("---")
            st.subheader(f"Top {embedding_method} Weighted Terms by Label")
            top_n = st.slider("Number of top terms", 10, 50, 20, 5)
            fig = self.viz["plot_top_weighted_terms"](
                matrix, vectorizer, df, label_col, method_name=embedding_method, top_n=top_n
            )
            st.pyplot(fig)
            plt.close(fig)

        st.divider()

    def show_error(self, message: str):
        """Display error message"""
        st.error(message)

    def show_success(self, message: str):
        """Display success message"""
        st.success(message)

    def show_loading(self, message: str = "Loading data..."):
        """Display loading spinner"""
        return st.spinner(message)

    def show_info(self, message: str):
        """Display info message"""
        st.info(message)
