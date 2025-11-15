import os
import re
import unicodedata

import pandas as pd
from dotenv import load_dotenv

from src import VnCoreNLP_Singleton

load_dotenv()

VNCORENLP_PATH = os.getenv("VNCORENLP_PATH")
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


def word_segmentation(text: str, lib: str = ""):
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
        rdrsegmenter = VnCoreNLP_Singleton.get_instance(VNCORENLP_PATH)
        text = rdrsegmenter.word_segment(text)
        text = " ".join(text)
        return text


def normalize_and_clean_vietnamese_text(df, text_column, lib):
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
        .apply(lambda x: word_segmentation(x, lib))
    )
    return df
