import re
import unicodedata

import pandas as pd
import py_vncorenlp

# def normalize_vietnamese(text):
#     # Normalize Unicode characters
#     text = unicodedata.normalize("NFC", text)
#     # Remove diacritics
#     text = "".join(
#         c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
#     )
#     return text


def clean_text(text):
    # # Convert to lowercase
    # text = text.lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)
    # Remove special characters, keep numbers
    text = re.sub(r"[^\w\s\d]", "", text)
    # Remove double white space
    text = re.sub(r"\s+", " ", text).strip()
    return text


def word_segmentation(text: str, lib: str = "pyvi"):
    """Words segmentation

    Args:
        text (str): Input text
        lib (str, optional): Segmentation libraries, including: pyvi, vncorenlp. Defaults to "pyvi".
    """
    if lib == "pyvi":
        from pyvi import ViTokenizer

        text = ViTokenizer.tokenize(text)
        return text
    else:
        rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir="vncorenlp")
        text = rdrsegmenter.word_segment(text)
        text = " ".join(text)
        return text


def normalize_and_clean_vietnamese_text(df, text_column):
    """
    Normalize and clean Vietnamese textual data in a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing the text data.
        text_column (str): The name of the column containing the text data.

    Returns:
        pd.DataFrame: A DataFrame with the cleaned and normalized text.
    """

    # Apply normalization and cleaning
    df[f"normalized_{text_column}"] = (
        df[text_column]
        .astype(str)
        # .apply(normalize_vietnamese)
        .apply(clean_text)
        .apply(lambda x: word_segmentation(x, "pyvi"))
    )
    return df
