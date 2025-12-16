import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, List, Optional, Any


class DetectionView:
    """View class for fake news detection page."""

    def render_title(self):
        """Render page title."""
        st.title("Fake News Detection")
        st.markdown("""
        Enter a news article text below to check if it's likely to be fake or real news.
        The system uses BM25 similarity to find the most similar articles in our database
        and predicts based on their labels.
        """)

    def render_input_form(self) -> Optional[str]:
        """Render text input form and return submitted text."""
        st.subheader("Input Document")

        text_input = st.text_area(
            "Paste your news article here:",
            height=200,
            placeholder="Enter the news article content you want to check...",
            key="detection_text_input"
        )

        col1, col2 = st.columns([1, 5])
        with col1:
            submit_button = st.button(
                "Detect",
                type="primary",
                use_container_width=True
            )

        if submit_button and text_input.strip():
            return text_input.strip()
        elif submit_button and not text_input.strip():
            st.warning("Please enter some text to analyze.")

        return None

    def render_result(self, result: Dict[str, Any]):
        """Render detection result with visualizations."""
        st.divider()
        st.subheader("Detection Result")

        prediction = result.get("prediction", "Unknown")
        confidence = result.get("confidence", 0.0)
        fake_probability = result.get("fake_probability", 0.5)

        # Main result display
        col1, col2, col3 = st.columns(3)

        with col1:
            if prediction == "FAKE":
                st.error(f"**Prediction: {prediction}**")
            else:
                st.success(f"**Prediction: {prediction}**")

        with col2:
            st.metric("Confidence", f"{confidence:.1%}")

        with col3:
            st.metric("Fake Probability", f"{fake_probability:.1%}")

        # Gauge chart for probability
        self._render_probability_gauge(fake_probability)

        # Similar articles
        if "similar_articles" in result and result["similar_articles"]:
            self._render_similar_articles(result["similar_articles"])

    def _render_probability_gauge(self, fake_probability: float):
        """Render a gauge chart showing fake news probability."""
        st.subheader("Fake News Probability")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fake_probability * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Fake News Score", 'font': {'size': 20}},
            number={'suffix': "%", 'font': {'size': 40}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "darkgray"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': '#2ecc71'},
                    {'range': [30, 70], 'color': '#f39c12'},
                    {'range': [70, 100], 'color': '#e74c3c'}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': fake_probability * 100
                }
            }
        ))

        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=50, b=20)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Interpretation
        if fake_probability < 0.3:
            st.info("**Low risk**: This article appears to be reliable based on similar content in our database.")
        elif fake_probability < 0.7:
            st.warning("**Medium risk**: This article shows mixed signals. Please verify from multiple sources.")
        else:
            st.error("**High risk**: This article shows patterns similar to known fake news. Exercise caution.")

    def _render_similar_articles(self, articles: List[Dict]):
        """Render table of similar articles used for prediction."""
        st.subheader("Similar Articles in Database")
        st.markdown("These are the most similar articles found in our database that were used for the prediction:")

        df = pd.DataFrame(articles)

        # Format the dataframe for display
        display_df = df[["rank", "title", "source", "label", "similarity_score"]].copy()
        display_df.columns = ["Rank", "Title", "Source", "Label", "Similarity"]
        display_df["Similarity"] = display_df["Similarity"].apply(lambda x: f"{x:.3f}")
        display_df["Label"] = display_df["Label"].apply(lambda x: "FAKE" if x == 1 else "REAL")

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        # Show label distribution
        fake_count = sum(1 for a in articles if a["label"] == 1)
        real_count = len(articles) - fake_count

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Similar Fake Articles", fake_count)
        with col2:
            st.metric("Similar Real Articles", real_count)

    def render_error(self, message: str):
        """Render error message."""
        st.error(f"Error: {message}")

    def render_loading(self):
        """Return a spinner context manager for loading states."""
        return st.spinner("Analyzing document...")

    def render_no_data_warning(self):
        """Render warning when no labeled data is available."""
        st.warning("""
        **No labeled data available for detection.**

        The detection system requires labeled articles (fake/real) in the database
        to make predictions. Please ensure the database contains labeled articles.
        """)
