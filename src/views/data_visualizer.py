import os
from threading import RLock
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import seaborn as sns
import umap
from rank_bm25 import BM25Okapi
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


def compute_bm25_embeddings(df: pd.DataFrame, text_col: str, max_features: int = None) -> Tuple:
    """Compute BM25 embeddings for text data

    Args:
        df (pd.DataFrame): DataFrame with text data
        text_col (str): Column name containing text
        max_features (int, optional): Maximum number of features. Defaults to None.

    Returns:
        bm25_matrix: BM25 sparse matrix
        vectorizer: Fitted CountVectorizer (for feature names)
    """
    # Auto-calculate max_features if not provided
    if max_features is None:
        avg_tokens = df[text_col].astype(str).str.split().str.len().mean()
        max_features = min(3000, int(avg_tokens * 1.5))
        print(
            f"Auto-selected max_features: {max_features} (based on avg tokens: {int(avg_tokens)})"
        )

    # First, use CountVectorizer to get vocabulary
    vectorizer = CountVectorizer(
        max_features=max_features, ngram_range=(1, 2), min_df=2, max_df=0.95, stop_words=None
    )

    # Fit vectorizer to get vocabulary
    vectorizer.fit(df[text_col].astype(str))
    vocabulary = vectorizer.vocabulary_
    feature_names = vectorizer.get_feature_names_out()

    # Tokenize documents according to vocabulary
    def tokenize_with_vocab(text):
        tokens = text.lower().split()
        return [token for token in tokens if token in vocabulary]

    # Tokenize all documents
    tokenized_corpus = [tokenize_with_vocab(doc) for doc in df[text_col].astype(str)]

    # Compute BM25
    bm25 = BM25Okapi(tokenized_corpus)

    # Create Matrix
    bm25_scores = []

    for doc_tokens in tokenized_corpus:
        # Get BM25 scores for this document
        doc_scores = bm25.get_scores(doc_tokens)
        bm25_scores.append(doc_scores)

    # Convert to sparse matrix format (similar to TF-IDF)
    bm25_matrix = sp.csr_matrix(bm25_scores)

    # Print info
    sparsity = 1.0 - (bm25_matrix.nnz / (bm25_matrix.shape[0] * bm25_matrix.shape[1]))
    print(f"BM25 matrix shape: {bm25_matrix.shape}")
    print(f"Sparsity: {sparsity*100:.2f}%")

    return bm25_matrix, vectorizer


def compute_bm25_normalized_embeddings(df: pd.DataFrame, text_col: str, max_features: int = None):
    """
    Compute BM25 embeddings using a simpler, more efficient approach

    Args:
        df: DataFrame with text data
        text_col: Column name containing text
        max_features: Maximum number of features

    Returns:
        bm25_matrix: BM25 sparse matrix
        vectorizer: Fitted CountVectorizer
    """
    from sklearn.feature_extraction.text import CountVectorizer

    # Auto-calculate max_features if not provided
    if max_features is None:
        avg_tokens = df[text_col].astype(str).str.split().str.len().mean()
        max_features = min(3000, int(avg_tokens * 1.5))

    # Use CountVectorizer to create term-document matrix
    vectorizer = CountVectorizer(
        max_features=max_features, ngram_range=(1, 2), min_df=2, max_df=0.95, stop_words=None
    )

    # Get term frequencies
    tf_matrix = vectorizer.fit_transform(df[text_col].astype(str))

    # BM25 parameters
    k1 = 1.5  # Term frequency saturation parameter
    b = 0.75  # Length normalization parameter

    # Calculate document lengths
    doc_lens = np.array(tf_matrix.sum(axis=1)).flatten()
    avg_doc_len = doc_lens.mean()

    # Calculate IDF (inverse document frequency)
    N = tf_matrix.shape[0]  # Number of documents
    df_terms = np.array((tf_matrix > 0).sum(axis=0)).flatten()  # Document frequency per term
    idf = np.log((N - df_terms + 0.5) / (df_terms + 0.5) + 1)

    # Calculate BM25 scores
    # BM25(q,d) = IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * |d| / avgdl))

    # Convert to dense for calculation (or use sparse operations)
    tf_dense = tf_matrix.toarray()

    # Length normalization factor
    len_norm = 1 - b + b * (doc_lens / avg_doc_len).reshape(-1, 1)

    # BM25 formula
    numerator = tf_dense * (k1 + 1)
    denominator = tf_dense + k1 * len_norm
    bm25_scores = idf * (numerator / denominator)

    # Convert back to sparse matrix
    bm25_matrix = sp.csr_matrix(bm25_scores)

    # Print info
    sparsity = 1.0 - (bm25_matrix.nnz / (bm25_matrix.shape[0] * bm25_matrix.shape[1]))
    print(f"BM25 matrix shape: {bm25_matrix.shape}")
    print(f"Sparsity: {sparsity*100:.2f}%")
    print(f"BM25 parameters: k1={k1}, b={b}")

    return bm25_matrix, vectorizer


def compute_embeddings(
    df: pd.DataFrame, text_col: str, method: str = "tfidf", max_features: int = None
):
    """
    Unified function to compute embeddings using TF-IDF or BM25

    Args:
        df: DataFrame with text data
        text_col: Column name containing text
        method: 'tfidf' or 'bm25'
        max_features: Maximum number of features

    Returns:
        matrix: Embedding matrix (sparse)
        vectorizer: Fitted vectorizer
    """
    if method.lower() == "tfidf":
        return compute_tfidf_embeddings(df, text_col, max_features)
    elif method.lower() == "bm25":
        return compute_bm25_normalized_embeddings(df, text_col, max_features)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'tfidf' or 'bm25'")


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


def plot_embedding_scatter(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    label_col: str,
    reduction_method: str = "PCA",
    embedding_method: str = "TF-IDF",
    title: str = None,
    figsize: tuple = (15, 10),
):
    """
    Create scatter plot of embeddings colored by label (works for both TF-IDF and BM25)
    """
    with plot_lock:
        fig, ax = plt.subplots(figsize=figsize)

        labels = df[label_col].unique()
        colors = ["#FF6B6B", "#4ECDC4", "#95E1D3"]
        label_names = {0: "Real News", 1: "Fake News", -1: "Unlabeled"}

        for idx, label in enumerate(labels):
            mask = (df[label_col] == label).values
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

        ax.set_xlabel(f"{reduction_method} Component 1", fontsize=12)
        ax.set_ylabel(f"{reduction_method} Component 2", fontsize=12)

        if title:
            ax.set_title(title, fontsize=16, fontweight="bold")
        else:
            ax.set_title(
                f"{embedding_method} Embeddings Visualization ({reduction_method})",
                fontsize=16,
                fontweight="bold",
            )

        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

    return fig


def plot_embedding_scatter_with_sources(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    source_col: str,
    label_col: str,
    reduction_method: str = "PCA",
    embedding_method: str = "TF-IDF",
    title: str = None,
    figsize: tuple = (15, 10),
):
    """
    Create scatter plot with source and label information
    """
    with plot_lock:
        fig, ax = plt.subplots(figsize=figsize)

        sources = df[source_col].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(sources)))

        for idx, source in enumerate(sources):
            source_mask = (df[source_col] == source).values

            for label in [0, 1]:
                label_mask = (df[label_col] == label).values
                mask = source_mask & label_mask

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

        ax.set_xlabel(f"{reduction_method} Component 1", fontsize=12)
        ax.set_ylabel(f"{reduction_method} Component 2", fontsize=12)

        if title:
            ax.set_title(title, fontsize=16, fontweight="bold")
        else:
            ax.set_title(
                f"{embedding_method} Embeddings by Source and Label ({reduction_method})",
                fontsize=16,
                fontweight="bold",
            )

        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

    return fig


def plot_embedding_3d(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    label_col: str,
    reduction_method: str = "PCA",
    embedding_method: str = "TF-IDF",
    title: str = None,
    figsize: tuple = (15, 12),
):
    """
    Create 3D scatter plot of embeddings (works for both TF-IDF and BM25)

    Args:
        df: DataFrame with labels
        embeddings: 3D embeddings array
        label_col: Column name for labels
        reduction_method: Name of dimensionality reduction method
        embedding_method: 'TF-IDF' or 'BM25'
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
            # Convert to numpy array for proper indexing
            mask = (df[label_col] == label).values
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

        ax.set_xlabel(f"{reduction_method} Component 1", fontsize=12)
        ax.set_ylabel(f"{reduction_method} Component 2", fontsize=12)
        ax.set_zlabel(f"{reduction_method} Component 3", fontsize=12)

        if title:
            ax.set_title(title, fontsize=16, fontweight="bold")
        else:
            ax.set_title(
                f"{embedding_method} 3D Embeddings ({reduction_method})",
                fontsize=16,
                fontweight="bold",
            )

        ax.legend(loc="best", fontsize=10)

        # Add grid
        ax.grid(True, alpha=0.3)

        # Improve viewing angle
        ax.view_init(elev=20, azim=45)

        plt.tight_layout()

    return fig


def plot_embedding_3d_with_sources(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    source_col: str,
    label_col: str,
    reduction_method: str = "PCA",
    embedding_method: str = "TF-IDF",
    title: str = None,
    figsize: tuple = (15, 12),
):
    """
    Create 3D scatter plot with source and label information

    Args:
        df: DataFrame with labels and sources
        embeddings: 3D embeddings array
        source_col: Column name for sources
        label_col: Column name for labels
        reduction_method: Name of dimensionality reduction method
        embedding_method: 'TF-IDF' or 'BM25'
        title: Plot title
        figsize: Figure size

    Returns:
        fig: Matplotlib figure
    """
    from mpl_toolkits.mplot3d import Axes3D

    with plot_lock:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        # Get unique sources
        sources = df[source_col].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(sources)))

        # Plot each source with different color
        for idx, source in enumerate(sources):
            source_mask = (df[source_col] == source).values

            # Separate by label within source
            for label in [0, 1]:
                label_mask = (df[label_col] == label).values
                mask = source_mask & label_mask

                if mask.sum() > 0:
                    marker = "o" if label == 0 else "^"
                    label_name = "Real" if label == 0 else "Fake"
                    ax.scatter(
                        embeddings[mask, 0],
                        embeddings[mask, 1],
                        embeddings[mask, 2],
                        c=[colors[idx]],
                        marker=marker,
                        label=f"{source} ({label_name})",
                        alpha=0.6,
                        s=80,
                        edgecolors="black",
                        linewidth=0.5,
                    )

        ax.set_xlabel(f"{reduction_method} Component 1", fontsize=12)
        ax.set_ylabel(f"{reduction_method} Component 2", fontsize=12)
        ax.set_zlabel(f"{reduction_method} Component 3", fontsize=12)

        if title:
            ax.set_title(title, fontsize=16, fontweight="bold")
        else:
            ax.set_title(
                f"{embedding_method} 3D Embeddings by Source and Label ({reduction_method})",
                fontsize=16,
                fontweight="bold",
            )

        # Place legend outside plot
        ax.legend(bbox_to_anchor=(1.15, 1), loc="upper left", fontsize=8)

        # Add grid
        ax.grid(True, alpha=0.3)

        # Improve viewing angle
        ax.view_init(elev=20, azim=45)

        plt.tight_layout()

    return fig


def plot_top_weighted_terms(
    matrix,
    vectorizer,
    df: pd.DataFrame,
    label_col: str,
    method_name: str = "TF-IDF",
    top_n: int = 20,
    figsize: tuple = (15, 10),
):
    """
    Plot top weighted terms for each label (works for both TF-IDF and BM25)
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
            mask = (df[label_col] == label).values
            mask_indices = np.where(mask)[0]

            # Average scores for this label
            label_scores = matrix[mask_indices].mean(axis=0).A1
            top_indices = label_scores.argsort()[-top_n:][::-1]
            top_terms = [feature_names[i] for i in top_indices]
            top_scores = [label_scores[i] for i in top_indices]

            axes[idx].barh(range(top_n), top_scores[::-1], color="skyblue")
            axes[idx].set_yticks(range(top_n))
            axes[idx].set_yticklabels(top_terms[::-1])
            axes[idx].set_xlabel(f"Average {method_name} Score")
            axes[idx].set_title(f'Top {top_n} Terms - {label_names.get(label, f"Label {label}")}')
            axes[idx].invert_yaxis()

        plt.tight_layout()

    return fig
