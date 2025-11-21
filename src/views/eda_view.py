from datetime import date, datetime
from typing import Any, Dict, List

import streamlit as st
from matplotlib import pyplot as plt

from src.views.data_visualizer import (
    generate_wordcloud,
    ngram_by_label,
    plot_length_distribution,
    plot_source_label_counts,
    plot_source_label_counts_grouped,
    plot_top_ngrams,
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
