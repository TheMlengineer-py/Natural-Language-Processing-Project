"""
downloader.py

 This module downloads the BBC dataset zip file from UCD and extracts it into the local `data/` directory.
It avoids redownloading if the dataset already exists.
"""

import os
import requests
import zipfile

BBC_DATASET_URL = "http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip"
DATA_RAW_PATH = "data"


def download_bbc_dataset():
    """
    Downloads and extracts the BBC dataset if not already present.
    Returns the path to the extracted 'bbc' folder.
    """
    os.makedirs(DATA_RAW_PATH, exist_ok=True)
    zip_path = os.path.join(DATA_RAW_PATH, "bbc.zip")
    extracted_dir = os.path.join(DATA_RAW_PATH, "bbc")

    if not os.path.exists(extracted_dir):
        print("[INFO] Downloading BBC dataset...")
        response = requests.get(BBC_DATASET_URL)
        with open(zip_path, "wb") as f:
            f.write(response.content)

        print("[INFO] Extracting dataset...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(DATA_RAW_PATH)

        if not os.path.exists(extracted_dir):
            raise FileNotFoundError("[ERROR] 'bbc' folder not found after extraction.")
        print("[DONE] Dataset downloaded and extracted.")
    else:
        print("[OK] BBC dataset already available.")

    return extracted_dir
