from threading import RLock

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
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
    # stop = set(STOPWORDS)
    labels = df[label_col].unique()
    for lbl in labels:
        subset = df[df[label_col] == lbl]
        text = " ".join(subset[text_col].astype(str).tolist())
        figs[lbl] = generate_wordcloud(text, title=f"WordCloud Label = {lbl}", stopwords=set())
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
            ax.bar(x + i * width - width/len(labels), counts[label], width, label=label)

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
