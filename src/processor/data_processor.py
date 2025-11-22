import os
import re

import pandas as pd
from dotenv import load_dotenv
from pyvi import ViTokenizer

from src.utils.vncorenlp_singleton import VnCoreNLP_Singleton

load_dotenv()


def clean_text(text: str):
    # # Convert to lowercase
    # text = text.lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)
    # Remove special characters, keep numbers
    text = re.sub(r"[^\w\s\d]", "", text)
    # Remove double white space
    text = re.sub(r"\s+", " ", text).strip()
    return text


def word_segmentation(text: str, lib: str = "", remove_stopwords: bool = False):
    """Words segmentation

    Args:
        text (str): Input text
        lib (str, optional): Segmentation libraries, including: pyvi, vncorenlp. Defaults to "pyvi".
    """
    if lib == "pyvi":
        text = ViTokenizer.tokenize(text)
        text = text.split(" ")
    else:
        VNCORENLP_PATH = os.getenv("VNCORENLP_PATH")
        rdrsegmenter = VnCoreNLP_Singleton.get_instance(VNCORENLP_PATH, ["wseg"])
        text = rdrsegmenter.word_segment(text)
        new_text = []
        for sentence in text:
            for w in sentence.split(" "):
                new_text.append(w)
        text = new_text

    if remove_stopwords:
        stopwords_path = os.getenv("STOPWORDS_PATH", "")
        with open(stopwords_path, "r", encoding="utf-8") as file:
            stopwords = set(w.strip() for w in file.readlines())
        clean_text = [w for w in text if w not in stopwords]
        print(clean_text)
        text = clean_text

    return " ".join(text)


def normalize_and_clean_vietnamese_text(df: pd.DataFrame, text_column: str, lib: str = ""):
    """
    Normalize and clean Vietnamese textual data in a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing the text data.
        text_column (str): The name of the column containing the text data.
        lib (str, optional): Segmentation libraries, including: pyvi, vncorenlp. Defaults to "pyvi".

    Returns:
        pd.DataFrame: A DataFrame with the cleaned and normalized text.
    """

    # Apply normalization and cleaning
    df[f"normalized_{text_column}"] = (
        df[text_column]
        .astype(str)
        # .apply(normalize_vietnamese)
        .apply(clean_text)
        .apply(lambda x: word_segmentation(x, lib, True))
    )
    return df


def basic_statistics(
    df: pd.DataFrame,
    label_col: str = "is_fake",
    source_col: str = None,
    date_col: str = None,
    text_col: str = "content",
) -> pd.DataFrame:
    """Generate and print basic statistics for a given DataFrame, including label counts,
    source counts, date distribution, and text length statistics. Optionally returns
    the DataFrame with additional columns for text length in words and characters.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        label_col (str, optional): The column name for the label (default is "is_fake").
        source_col (str, optional): The column name for the source (default is None).
        date_col (str, optional): The column name for the date (default is None).
        text_col (str, optional): The column name for the text content (default is "content").

    Returns:
        pd.DataFrame: The input DataFrame with additional columns:
            - "text_len_words": Length of text in words.
            - "text_len_characters": Length of text in characters.

    Notes:
        - If `source_col` is provided, the function will print the count of unique sources.
        - If `date_col` is provided, the function will convert the column to datetime
          and print the distribution of dates.
        - The function assumes that `text_col` contains text data and calculates
          statistics based on its content.
    """
    print("Basic Stats")
    print(f"Shape: {df.shape}")  # Shape of dataset (rows/columns)
    print("\nLabel counts:")
    print(df[label_col].value_counts())
    if source_col:
        print("\nSource count:")
        print(df[source_col].value_counts())
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        print("\nDate distribution (by day):")
        print(df[date_col].dt.date.value_counts().sort_index())
    df["text_len_words"] = df[text_col].astype(str).apply(lambda x: len(x.split()))
    df["text_len_characters"] = df[text_col].astype(str).apply(len)
    print("\nText length (words) stats:\n", df["text_len_words"].describe())
    print("\nText length (chars) stats:\n", df["text_len_characters"].describe())
    return df
