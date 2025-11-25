from typing import Any, Dict, Tuple

import pandas as pd

from src.data_models.health_news_model import HealthNewsModel


class EDAController:
    """Controller layer - coordinates between model and view"""

    def __init__(self, model: HealthNewsModel):
        self.model = model
        self.config = {
            "label_col": "is_fake",
            "source_col": "source",
            "date_col": "crawled_at",
            "text_col": "normalized_content",
        }
        self.filters = {}

    def set_filters(self, filters: Dict[str, Any]):
        """Set filters for data loading"""
        self.filters = filters

    def initialize_data(self) -> None:
        """Load and prepare data"""
        self.model.load_data(filters=self.filters if self.filters else None)
        self.model.prepare_for_analysis(text_col=self.config["text_col"])

    def get_metrics_data(self) -> Dict[str, int]:
        """Get metrics for display"""
        return self.model.get_metrics()

    def get_filter_options(self) -> Dict[str, Any]:
        """Get available filter options"""
        sources = self.model.get_available_sources()
        date_range = self.model.get_available_date_range()

        return {"sources": sources, "min_date": date_range[0], "max_date": date_range[1]}

    def get_length_distribution_data(self) -> Tuple[pd.DataFrame, str]:
        """Prepare data for length distribution plot"""
        df = self.model.get_dataframe()
        return df, self.config["label_col"]

    def get_wordcloud_data(self) -> Dict[str, Any]:
        """Prepare data for word clouds"""
        text_all = self.model.get_all_text(self.config["text_col"])

        text_by_label = {}
        df = self.model.get_dataframe()
        unique_labels = df[self.config["label_col"]].unique()

        for label in unique_labels:
            if label != -1:  # Skip unlabeled data
                text_by_label[label] = self.model.get_text_by_label(
                    self.config["text_col"], self.config["label_col"], label
                )

        return {"all": text_all, "by_label": text_by_label}

    def get_ngrams_data(self) -> Tuple[pd.DataFrame, str]:
        """Prepare data for n-grams analysis"""
        df = self.model.get_dataframe()
        return df, self.config["text_col"]

    def get_source_label_data(self) -> Tuple[pd.DataFrame, str, str]:
        """Prepare data for source-label analysis"""
        df = self.model.get_dataframe()
        return df, self.config["source_col"], self.config["label_col"]

    def prepare_tfidf_embedding_data(
        self, method: str = "pca", n_components: int = 2, max_features: int = 5000
    ) -> Tuple[pd.DataFrame, str, int, int]:
        """Prepare TF-IDF embeddings for visualization

        Args:
            method: Dimensionality reduction method ('pca', 'tsne', 'umap', 'svd')
            n_components: Number of dimensions (2 or 3)
            max_features: Maximum TF-IDF features

        Returns:
            Tuple of (df, embeddings, method_name, tfidf_matrix, vectorizer)
        """
        df = self.model.get_dataframe()
        text_col = self.config["text_col"]

        # This will be computed in the view using visualizer functions
        return df, text_col, method, n_components, max_features
