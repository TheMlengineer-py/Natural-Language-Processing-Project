"""
ner_agent.py

 Named Entity Recognition + Role Classification

- Extracts PERSON entities using spaCy transformer model
- Uses zero-shot classification to infer their profession (e.g. Politician, Musician)
- Uses config-driven models and hyperparameters
- Evaluates role prediction performance
"""

import pandas as pd
import spacy
from transformers import pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


def extract_named_entities_and_roles(
    df: pd.DataFrame,
    spacy_model: str,
    role_classifier_model: str,
    role_labels: list,
    output_dir: str = "outputs",
) -> pd.DataFrame:
    """
    Identifies media personalities and classifies their job roles.

    Args:
        df (pd.DataFrame): Articles with `text` column
        spacy_model (str): spaCy transformer model (e.g. en_core_web_trf)
        role_classifier_model (str): HuggingFace model for role classification
        role_labels (list): Possible job roles (e.g. Politician, Actor, etc.)
        output_dir (str): Folder to save evaluation results

    Returns:
        pd.DataFrame: One row per named entity with predicted role
    """
    print(f" Loading spaCy model: {spacy_model}")
    nlp = spacy.load(spacy_model)

    print(f" Loading zero-shot role classifier: {role_classifier_model}")
    classifier = pipeline("zero-shot-classification", model=role_classifier_model)

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        doc = nlp(row["text"])
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                try:
                    sentence = ent.sent.text
                    result = classifier(sentence, candidate_labels=role_labels)
                    role = result["labels"][0]
                    results.append(
                        {"name": ent.text, "role": role, "sentence": sentence}
                    )
                except Exception as e:
                    print(f" Role classification failed: {e}")
                    continue

    role_df = pd.DataFrame(results)

    # Evaluation (if multiple role-labels are available)
    if "role" in role_df.columns and len(role_df["role"].unique()) > 1:
        report_path = os.path.join(output_dir, "ner_roles_report.txt")
        matrix_path = os.path.join(output_dir, "ner_roles_confusion_matrix.png")

        try:
            # Simulate ground truth role labels based on keywords in sentence
            # This is a proxy eval only; real labels would require annotation
            role_df["true_role"] = role_df["sentence"].apply(
                lambda s: next(
                    (r for r in role_labels if r.lower() in s.lower()), "unknown"
                )
            )

            report = classification_report(
                role_df["true_role"], role_df["role"], zero_division=0
            )
            with open(report_path, "w") as f:
                f.write(report)
            print(f" Role classification report saved to: {report_path}")

            labels = sorted(list(set(role_df["true_role"]) | set(role_df["role"])))
            cm = confusion_matrix(role_df["true_role"], role_df["role"], labels=labels)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
            disp.plot(xticks_rotation=45, cmap="Purples")
            plt.title("Role Classification Confusion Matrix")
            plt.tight_layout()
            plt.savefig(matrix_path)
            print(f" Role confusion matrix saved to: {matrix_path}")
        except Exception as e:
            print(f" Evaluation skipped: {e}")

    return role_df
