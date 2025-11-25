from datetime import date, datetime
from typing import Any, Dict, List

import streamlit as st
from matplotlib import pyplot as plt

from src.views.data_visualizer import (
    compute_tfidf_embeddings,
    generate_wordcloud,
    ngram_by_label,
    plot_length_distribution,
    plot_source_label_counts,
    plot_source_label_counts_grouped,
    plot_tfidf_3d,
    plot_tfidf_scatter,
    plot_tfidf_scatter_with_sources,
    plot_top_ngrams,
    plot_top_tfidf_terms,
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
            "compute_tfidf_embeddings": compute_tfidf_embeddings,
            "plot_tfidf_scatter_with_sources": plot_tfidf_scatter_with_sources,
            "plot_tfidf_scatter": plot_tfidf_scatter,
            "plot_tfidf_3d": plot_tfidf_3d,
            "plot_top_tfidf_terms": plot_top_tfidf_terms,
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
        """Render TF-IDF embeddings visualization section"""
        st.header("TF-IDF Embeddings Visualization")

        # Settings in sidebar
        with st.sidebar:
            st.subheader("TF-IDF Settings")
            method = st.selectbox(
                "Reduction Method",
                options=["PCA", "t-SNE", "UMAP", "SVD"],
                index=0,
                help="Dimensionality reduction method for visualization",
            )

            n_components = st.radio(
                "Dimensions", options=[2, 3], index=0, help="2D or 3D visualization"
            )

            max_features = st.slider(
                "Max TF-IDF Features",
                min_value=1000,
                max_value=10000,
                value=5000,
                step=1000,
                help="Maximum number of TF-IDF features",
            )

            show_sources = st.checkbox(
                "Color by Source", value=False, help="Show sources in addition to labels"
            )

            show_top_terms = st.checkbox(
                "Show Top TF-IDF Terms", value=True, help="Display top TF-IDF terms for each label"
            )

        # Compute embeddings
        with st.spinner(f"Computing TF-IDF embeddings using {method}..."):
            # Compute TF-IDF
            tfidf_matrix, vectorizer = self.viz["compute_tfidf_embeddings"](
                df, text_col, max_features=max_features
            )

            # Reduce dimensions
            embeddings = self.viz["reduce_dimensions"](
                tfidf_matrix, method=method.lower().replace("-", ""), n_components=n_components
            )

        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Documents", len(df))
        with col2:
            st.metric("TF-IDF Features", tfidf_matrix.shape[1])
        with col3:
            st.metric("Embedding Dimensions", n_components)

        st.divider()

        # Plot embeddings
        if n_components == 2:
            if show_sources:
                st.subheader(f"2D {method} Embeddings - By Source and Label")
                fig = self.viz["plot_tfidf_scatter_with_sources"](
                    df, embeddings, source_col, label_col, method=method
                )
            else:
                st.subheader(f"2D {method} Embeddings - By Label")
                fig = self.viz["plot_tfidf_scatter"](df, embeddings, label_col, method=method)
            st.pyplot(fig)
            plt.close(fig)
        else:  # 3D
            st.subheader(f"3D {method} Embeddings")
            fig = self.viz["plot_tfidf_3d"](df, embeddings, label_col, method=method)
            st.pyplot(fig)
            plt.close(fig)

        # Show top TF-IDF terms
        if show_top_terms:
            st.divider()
            st.subheader("Top TF-IDF Terms by Label")
            top_n = st.slider("Number of top terms", 10, 50, 20, 5)
            fig = self.viz["plot_top_tfidf_terms"](
                tfidf_matrix, vectorizer, df, label_col, top_n=top_n
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
