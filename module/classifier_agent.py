"""
classifier_agent.py

 Sub-category classification using zero-shot LLM (e.g., BART or DistilBART).

- Classifies articles into sub-categories (stock market, football, cinema, etc.)
- Uses config for model selection and hyperparameters
- Generates evaluation report and confusion matrix
"""

import pandas as pd
from transformers import pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


# Predefined labels for each top-level category
SUBCATEGORY_LABELS = {
    "business": [
        "stock market",
        "mergers and acquisitions",
        "company news",
        "real estate",
        "finance",
    ],
    "entertainment": ["cinema", "music", "celebrity", "literature", "theatre"],
    "sport": ["football", "cricket", "olympics", "rugby", "tennis"],
    "politics": ["government", "elections", "policy", "foreign affairs"],
    "tech": ["AI", "gadgets", "cybersecurity", "software"],
}


def classify_subcategories(
    df: pd.DataFrame,
    model_name: str,
    top_k: int = 1,
    max_length: int = 1000,
    output_dir: str = "outputs",
) -> pd.DataFrame:
    """
    Applies zero-shot classification to predict sub-categories, and evaluates results.

    Args:
        df (pd.DataFrame): Articles with columns `category` and `text`
        model_name (str): HuggingFace zero-shot model
        top_k (int): How many top predictions to consider
        max_length (int): Truncate text for inference
        output_dir (str): Where to save report and confusion matrix

    Returns:
        pd.DataFrame: Original dataframe + `sub_category` column
    """
    print(f" Loading zero-shot classifier: {model_name}")
    classifier = pipeline("zero-shot-classification", model=model_name)

    predictions = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        top_cat = row["category"].lower()
        labels = SUBCATEGORY_LABELS.get(top_cat, [])
        if not labels:
            predictions.append("general")
            continue

        try:
            result = classifier(
                row["text"][:max_length],
                candidate_labels=labels,
                top_k=top_k,
                multi_label=False,
            )
            pred = (
                result["labels"][0] if isinstance(result, dict) else result[0]["label"]
            )
        except Exception as e:
            print(f"⚠ Skipped row due to error: {e}")
            pred = "error"

        predictions.append(pred)

    df["sub_category"] = predictions

    # Evaluate results
    report_path = os.path.join(output_dir, "classifier_report.txt")
    matrix_path = os.path.join(output_dir, "classifier_confusion_matrix.png")

    try:
        y_true = df["category"]
        y_pred = df["sub_category"]

        report = classification_report(y_true, y_pred, zero_division=0)
        with open(report_path, "w") as f:
            f.write(report)
        print(f" Classification report saved to: {report_path}")

        # Confusion matrix
        labels = sorted(list(set(y_true) | set(y_pred)))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(xticks_rotation=45, cmap="Blues")
        plt.tight_layout()
        plt.title("Sub-category Confusion Matrix")
        plt.savefig(matrix_path)
        print(f" Confusion matrix saved to: {matrix_path}")

    except Exception as e:
        print(f" Evaluation skipped due to: {e}")

    return df
