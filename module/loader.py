"""
loader.py

 This module reads the extracted BBC dataset folder structure and loads each article
into a Pandas DataFrame with two columns: `category` and `text`.
"""

import os
import pandas as pd


def load_bbc_data(bbc_folder_path: str) -> pd.DataFrame:
    """
    Loads the BBC dataset from the given 'bbc' folder into a DataFrame.

    Args:
        bbc_folder_path (str): Path to the `bbc` folder containing category subfolders.

    Returns:
        pd.DataFrame: DataFrame with columns `category` and `text`
    """
    records = []

    for category in os.listdir(bbc_folder_path):
        category_path = os.path.join(bbc_folder_path, category)

        if not os.path.isdir(category_path):
            continue  # Skip non-folder items like README.TXT

        for file_name in os.listdir(category_path):
            file_path = os.path.join(category_path, file_name)
            with open(file_path, encoding="latin1") as f:
                text = f.read().strip()
                records.append({"category": category, "text": text})

    return pd.DataFrame(records)
