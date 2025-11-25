import os
from threading import RLock
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.manifold import TSNE
from wordcloud import WordCloud

plot_lock = RLock()


def plot_length_distribution(df: pd.DataFrame, label_col: str):
    with plot_lock:
        fig, ax = plt.subplots(figsize=(15, 10))
        sns.histplot(data=df, x="word_count", hue=label_col, bins=50, kde=True, ax=ax)
        ax.set_title("Words Distribution by Labels")
        ax.set_xlabel("Number of Words")
        ax.set_ylabel("Number of News")
    return fig


def plot_source_label_counts(df: pd.DataFrame, source_col: str, label_col: str):
    with plot_lock:
        # Group by source and label
        counts = df.groupby([source_col, label_col]).size().unstack(fill_value=0)
        print("\nThe number of news by sources and labels:\n", counts)

        # Bar chart
        fig, ax = plt.subplots(figsize=(15, 10))
        counts.plot(kind="bar", ax=ax)
        ax.set_title("Number of news by sources and labels")
        ax.set_xlabel("Source")
        ax.set_ylabel("Number of news")
        ax.set_xticklabels(counts.index, rotation=45, ha="right")
        ax.legend(title=label_col)
        plt.tight_layout()
    return fig


def generate_wordcloud(text: str, title: str = None, stopwords: set = None):
    with plot_lock:
        wc = WordCloud(
            width=800,
            height=400,
            background_color="white",
            stopwords=stopwords or set(),
            collocations=False,
        ).generate(text)
        fig, ax = plt.subplots(figsize=(15, 7.5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        if title:
            ax.set_title(title, fontsize=20)
    return fig


def wordcloud_by_label(df: pd.DataFrame, text_col: str, label_col: str):
    figs = {}
    stopwords_path = os.getenv("vietnamese-stopwords-dash.txt")
    with open(stopwords_path, "r") as file:
        stopwords = set(w.strip() for w in file.readlines())
    labels = df[label_col].unique()
    for lbl in labels:
        subset = df[df[label_col] == lbl]
        text = " ".join(subset[text_col].astype(str).tolist())
        figs[lbl] = generate_wordcloud(
            text, title=f"WordCloud Label = {lbl}", stopwords=set(stopwords)
        )
    return figs


def plot_source_label_counts_grouped(df: pd.DataFrame, source_col: str, label_col: str):
    with plot_lock:
        # Group news by sources and labels
        counts = df.groupby([source_col, label_col]).size().unstack(fill_value=0)
        print("\nThe number of news by sources and labels:\n", counts)

        # Sources
        sources = counts.index.tolist()
        # Labels
        labels = counts.columns.tolist()

        # Create index for the bars
        x = np.arange(len(sources))
        width = 0.35

        fig, ax = plt.subplots(figsize=(15, 10))

        # First bar
        for i, label in enumerate(labels):
            ax.bar(x + i * width - width / len(labels), counts[label], width, label=label)

        # Set labels
        ax.set_xticks(x)
        ax.set_xticklabels(sources, rotation=45, ha="right")

        ax.set_xlabel("Source")
        ax.set_ylabel("Number of news")
        ax.set_title("Number of news by sources and labels")
        ax.legend(title=label_col)

        plt.tight_layout()
    return fig


def plot_top_ngrams(df: pd.DataFrame, text_col: str, ngram_range: tuple = (1, 1), top_n: int = 20):
    with plot_lock:
        vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words=None)
        X = vectorizer.fit_transform(df[text_col].astype(str))
        sum_words = X.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        top = words_freq[:top_n]
        words = [w for w, c in top]
        counts = [c for w, c in top]
        fig, ax = plt.subplots(figsize=(15, 10))
        sns.barplot(x=counts, y=words, ax=ax)
        ax.set_title(f"Top {top_n} n-gram (range = {ngram_range})")
        ax.set_xlabel("The appearance times")
        ax.set_ylabel("N-gram")
    return fig


def ngram_by_label(
    df: pd.DataFrame, text_col: str, label_col: str, ngram_range: tuple = (1, 1), top_n: int = 15
):
    figs = {}
    labels = df[label_col].unique()
    for lbl in labels:
        print(f"\n--- Labels: {lbl} ---")
        subset = df[df[label_col] == lbl]
        figs[lbl] = plot_top_ngrams(subset, text_col, ngram_range=ngram_range, top_n=top_n)
    return figs


def compute_tfidf_embeddings(df: pd.DataFrame, text_col: str, max_features: int = 2000) -> Tuple:
    """
    Compute TF-IDF embeddings for text data

    Args:
        df: DataFrame with text data
        text_col: Column name containing text
        max_features: Maximum number of features for TF-IDF

    Returns:
        tfidf_matrix: TF-IDF matrix
        vectorizer: Fitted TF-IDF vectorizer
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features, ngram_range=(1, 2), min_df=2, max_df=0.95, stop_words=None
    )
    tfidf_matrix = vectorizer.fit_transform(df[text_col].astype(str))
    return tfidf_matrix, vectorizer


def reduce_dimensions(
    tfidf_matrix, method: str = "pca", n_components: int = 2, random_state: int = 42
):
    """
    Reduce TF-IDF dimensions for visualization

    Args:
        tfidf_matrix: TF-IDF sparse matrix
        method: Reduction method ('pca', 'tsne', 'umap', 'svd')
        n_components: Number of dimensions (typically 2 or 3)
        random_state: Random seed for reproducibility

    Returns:
        embeddings: Reduced dimension embeddings
    """
    if method == "pca":
        reducer = PCA(n_components=n_components, random_state=random_state)
        embeddings = reducer.fit_transform(tfidf_matrix.toarray())
    elif method == "svd":
        reducer = TruncatedSVD(n_components=n_components, random_state=random_state)
        embeddings = reducer.fit_transform(tfidf_matrix)
    elif method == "tsne":
        # First reduce to 50 dimensions with SVD for efficiency
        if tfidf_matrix.shape[1] > 50:
            svd = TruncatedSVD(n_components=50, random_state=random_state)
            tfidf_reduced = svd.fit_transform(tfidf_matrix)
        else:
            tfidf_reduced = tfidf_matrix.toarray()
        reducer = TSNE(n_components=n_components, random_state=random_state, perplexity=30)
        embeddings = reducer.fit_transform(tfidf_reduced)
    elif method == "umap":
        reducer = umap.UMAP(n_components=n_components, random_state=random_state)
        embeddings = reducer.fit_transform(tfidf_matrix)
    else:
        raise ValueError(f"Unknown method: {method}")

    return embeddings


def plot_tfidf_scatter(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    label_col: str,
    method: str = "PCA",
    title: str = None,
    figsize: tuple = (15, 10),
):
    """
    Create scatter plot of TF-IDF embeddings colored by label

    Args:
        df: DataFrame with labels
        embeddings: 2D embeddings array
        label_col: Column name for labels
        method: Name of dimensionality reduction method
        title: Plot title
        figsize: Figure size

    Returns:
        fig: Matplotlib figure
    """
    with plot_lock:
        fig, ax = plt.subplots(figsize=figsize)

        # Get unique labels
        labels = df[label_col].unique()
        colors = [
            "#FF6B6B",
            "#4ECDC4",
            "#95E1D3",
        ]  # Red for fake, teal for real, green for unknown
        label_names = {0: "Real News", 1: "Fake News", -1: "Unlabeled"}

        # Plot each label with different color
        for idx, label in enumerate(labels):
            mask = df[label_col] == label
            ax.scatter(
                embeddings[mask, 0],
                embeddings[mask, 1],
                c=colors[idx % len(colors)],
                label=label_names.get(label, f"Label {label}"),
                alpha=0.6,
                s=50,
                edgecolors="black",
                linewidth=0.5,
            )

        ax.set_xlabel(f"{method} Component 1", fontsize=12)
        ax.set_ylabel(f"{method} Component 2", fontsize=12)

        if title:
            ax.set_title(title, fontsize=16, fontweight="bold")
        else:
            ax.set_title(
                f"TF-IDF Embeddings Visualization ({method})", fontsize=16, fontweight="bold"
            )

        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

    return fig


def plot_tfidf_scatter_with_sources(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    source_col: str,
    label_col: str,
    method: str = "PCA",
    title: str = None,
    figsize: tuple = (15, 10),
):
    """
    Create scatter plot of TF-IDF embeddings with source and label information

    Args:
        df: DataFrame with labels and sources
        embeddings: 2D embeddings array
        source_col: Column name for sources
        label_col: Column name for labels
        method: Name of dimensionality reduction method
        title: Plot title
        figsize: Figure size

    Returns:
        fig: Matplotlib figure
    """
    with plot_lock:
        fig, ax = plt.subplots(figsize=figsize)

        # Get unique sources
        sources = df[source_col].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(sources)))

        # Plot each source with different color
        for idx, source in enumerate(sources):
            source_mask = df[source_col] == source

            # Separate by label within source
            for label in [0, 1]:
                mask = source_mask & (df[label_col] == label)
                if mask.sum() > 0:
                    marker = "o" if label == 0 else "^"
                    label_name = "Real" if label == 0 else "Fake"
                    ax.scatter(
                        embeddings[mask, 0],
                        embeddings[mask, 1],
                        c=[colors[idx]],
                        marker=marker,
                        label=f"{source} ({label_name})",
                        alpha=0.6,
                        s=80,
                        edgecolors="black",
                        linewidth=0.5,
                    )

        ax.set_xlabel(f"{method} Component 1", fontsize=12)
        ax.set_ylabel(f"{method} Component 2", fontsize=12)

        if title:
            ax.set_title(title, fontsize=16, fontweight="bold")
        else:
            ax.set_title(
                f"TF-IDF Embeddings by Source and Label ({method})", fontsize=16, fontweight="bold"
            )

        # Place legend outside plot
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

    return fig


def plot_tfidf_3d(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    label_col: str,
    method: str = "PCA",
    title: str = None,
    figsize: tuple = (15, 12),
):
    """
    Create 3D scatter plot of TF-IDF embeddings

    Args:
        df: DataFrame with labels
        embeddings: 3D embeddings array
        label_col: Column name for labels
        method: Name of dimensionality reduction method
        title: Plot title
        figsize: Figure size

    Returns:
        fig: Matplotlib figure
    """
    from mpl_toolkits.mplot3d import Axes3D

    with plot_lock:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        # Get unique labels
        labels = df[label_col].unique()
        colors = ["#FF6B6B", "#4ECDC4", "#95E1D3"]
        label_names = {0: "Real News", 1: "Fake News", -1: "Unlabeled"}

        # Plot each label with different color
        for idx, label in enumerate(labels):
            mask = df[label_col] == label
            ax.scatter(
                embeddings[mask, 0],
                embeddings[mask, 1],
                embeddings[mask, 2],
                c=colors[idx % len(colors)],
                label=label_names.get(label, f"Label {label}"),
                alpha=0.6,
                s=50,
                edgecolors="black",
                linewidth=0.5,
            )

        ax.set_xlabel(f"{method} Component 1", fontsize=12)
        ax.set_ylabel(f"{method} Component 2", fontsize=12)
        ax.set_zlabel(f"{method} Component 3", fontsize=12)

        if title:
            ax.set_title(title, fontsize=16, fontweight="bold")
        else:
            ax.set_title(f"TF-IDF 3D Embeddings ({method})", fontsize=16, fontweight="bold")

        ax.legend(loc="best", fontsize=10)
        plt.tight_layout()

    return fig


def plot_top_tfidf_terms(
    tfidf_matrix,
    vectorizer,
    df: pd.DataFrame,
    label_col: str,
    top_n: int = 20,
    figsize: tuple = (15, 10),
):
    """
    Plot top TF-IDF terms for each label

    Args:
        tfidf_matrix: TF-IDF sparse matrix
        vectorizer: Fitted TF-IDF vectorizer
        df: DataFrame with labels
        label_col: Column name for labels
        top_n: Number of top terms to show
        figsize: Figure size

    Returns:
        fig: Matplotlib figure
    """
    with plot_lock:
        labels = sorted(df[label_col].unique())
        n_labels = len(labels)

        fig, axes = plt.subplots(1, n_labels, figsize=figsize)
        if n_labels == 1:
            axes = [axes]

        feature_names = vectorizer.get_feature_names_out()
        label_names = {0: "Real News", 1: "Fake News", -1: "Unlabeled"}

        for idx, label in enumerate(labels):
            # Convert boolean mask to numpy array of indices
            mask = (df[label_col] == label).values
            mask_indices = np.where(mask)[0]

            # Use integer indexing instead of boolean mask
            label_tfidf = tfidf_matrix[mask_indices].mean(axis=0).A1
            top_indices = label_tfidf.argsort()[-top_n:][::-1]
            top_terms = [feature_names[i] for i in top_indices]
            top_scores = [label_tfidf[i] for i in top_indices]

            axes[idx].barh(range(top_n), top_scores[::-1], color="skyblue")
            axes[idx].set_yticks(range(top_n))
            axes[idx].set_yticklabels(top_terms[::-1])
            axes[idx].set_xlabel("Average TF-IDF Score")
            axes[idx].set_title(f'Top {top_n} Terms - {label_names.get(label, f"Label {label}")}')
            axes[idx].invert_yaxis()

        plt.tight_layout()

    return fig
