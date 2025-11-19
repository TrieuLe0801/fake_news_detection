import os
import re

import pandas as pd
from dotenv import load_dotenv
from pyvi import ViTokenizer

from src import VnCoreNLP_Singleton

load_dotenv()

VNCORENLP_PATH = os.getenv("VNCORENLP_PATH")
rdrsegmenter = VnCoreNLP_Singleton.get_instance(VNCORENLP_PATH)


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
        text = ViTokenizer.tokenize(text)
        return text
    else:
        text = rdrsegmenter.word_segment(text)
        text = " ".join(text)
        return text


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
        .apply(lambda x: word_segmentation(x, lib))
    )
    return df
