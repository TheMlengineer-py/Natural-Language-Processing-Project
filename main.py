"""
main.py

Entry point for the full HMLR NLP pipeline.

Performs:

#     - Sub-category classification
#     - Named entity + role extraction
#     - April event summarisation
#     - Evaluation + statistical reporting
#     - Saves results into `outputs/` and `outputs/reports/`

"""

import os
import time
import csv
import logging
import pandas as pd

from module.downloader import download_bbc_dataset
from module.loader import load_bbc_data
from module.config_loader import load_config
from module.report_generator import generate_pipeline_report
from module.classifier_agent import classify_subcategories
from module.ner_agent import extract_named_entities_and_roles
from module.summarizer_agent import summarise_april_events

# Output directories
OUTPUT_DIR = "outputs"
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def run_pipeline():
    timings = {}
    config = load_config("config/pipeline.yaml")
    sample_size = config["pipeline"].get("sample_size", 200)

    logging.info(" HMLR NLP pipeline started")

    # Step 1: Load dataset
    logging.info(" Loading BBC dataset...")
    t0 = time.time()
    dataset_path = download_bbc_dataset()
    df = load_bbc_data(dataset_path)

    if sample_size > 0:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        logging.info(f" Using {sample_size} articles.")
    else:
        logging.info(f" Using full dataset of {len(df)} articles.")

    timings["Data Load"] = time.time() - t0

    # Step 2: Classifier
    t1 = time.time()
    classifier_cfg = config["classifier"]
    # model_name = classifier_cfg["fast_model"] if fast_mode else classifier_cfg["model"]
    model_name = classifier_cfg["model"]

    logging.info(f"Using classifier model: {model_name}")

    df = classify_subcategories(
        df,
        model_name=model_name,
        top_k=classifier_cfg.get("top_k", 1),
        max_length=classifier_cfg.get("max_length", 1000),
        output_dir=REPORTS_DIR,
    )
    df.to_csv(f"{OUTPUT_DIR}/bbc_subcategories.csv", index=False)
    timings["Classification"] = time.time() - t1

    # # Step 2: Reload classifier results
    # logging.info(" Skipping sub-category classification (already completed).")
    # df = pd.read_csv(os.path.join(OUTPUT_DIR, "bbc_subcategories.csv"))

    # Step 3: NER + Role
    t2 = time.time()
    ner_cfg = config["ner"]
    ner_df = extract_named_entities_and_roles(
        df,
        spacy_model=ner_cfg["spacy_model"],
        role_classifier_model=ner_cfg["role_classifier_model"],
        role_labels=ner_cfg["role_labels"],
        output_dir=REPORTS_DIR,
    )
    ner_df.to_csv(f"{OUTPUT_DIR}/ner_roles.csv", index=False)
    timings["NER + Role"] = time.time() - t2

    # Step 3: Reload NER results
    # logging.info(" Skipping named entity recognition (already completed).")
    # ner_df = pd.read_csv(os.path.join(OUTPUT_DIR, "ner_roles.csv"))

    #  Step 4: Summarisation
    t3 = time.time()
    summary_cfg = config["summarizer"]
    summary_df = summarise_april_events(
        df,
        model_name=summary_cfg["model"],
        max_length=summary_cfg["max_length"],
        min_length=summary_cfg["min_length"],
        do_sample=summary_cfg.get("do_sample", False),
        output_dir=REPORTS_DIR,
    )
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "april_summaries.csv"), index=False)
    timings["Summarisation"] = time.time() - t3

    # Final report
    generate_pipeline_report(
        timings=timings,
        sample_size=sample_size,
        output_dir=OUTPUT_DIR,
        report_dir=REPORTS_DIR,
        model_name=None,
        classifier_cfg=None,
        ner_cfg=config["ner"],
        summary_cfg=summary_cfg,
    )

    logging.info(" Summarisation completed and report generated.")


if __name__ == "__main__":
    run_pipeline()
