"""
report_generator.py

Generates the Markdown report, optional PDF export, and pipeline timing CSV
based on the outputs of the NLP pipeline.
"""

import os
import shutil
import subprocess
import csv


def generate_pipeline_report(
    timings: dict,
    sample_size: int,
    output_dir: str,
    report_dir: str,
    model_name: str = None,
    classifier_cfg: dict = None,
    ner_cfg: dict = None,
    summary_cfg: dict = None,
):
    """
    Generates a Markdown evaluation report and exports to PDF (if pandoc is installed),
    and saves runtime timing summary to CSV.

    Args:
        timings (dict): Dictionary of timing per step
        sample_size (int): How many rows were processed
        output_dir (str): Root output directory
        report_dir (str): Subdirectory where reports are saved
        model_name (str): Classifier model used (optional)
        classifier_cfg (dict): Classifier config (optional)
        ner_cfg (dict): NER agent config (required)
        summary_cfg (dict): Summariser config (required)
    """
    os.makedirs(report_dir, exist_ok=True)
    report_md = os.path.join(report_dir, "report.md")

    with open(report_md, "w") as f:
        f.write("# HMLR NLP Pipeline Evaluation Report\n\n")

        # Classifier Report
        subcat_csv = os.path.join(output_dir, "bbc_subcategories.csv")
        if os.path.exists(subcat_csv):
            f.write("## Classifier: Sub-Category Prediction\n")
            f.write(f"- **Model Used:** {model_name or '(unknown)'}\n")
            if classifier_cfg:
                f.write(f"- **Top-K:** {classifier_cfg.get('top_k', 1)}\n")
            f.write("- **Report:** [`classifier_report.txt`](classifier_report.txt)\n")
            f.write(
                "- **Confusion Matrix:** ![Confusion Matrix](classifier_confusion_matrix.png)\n\n"
            )

        # NER Report
        f.write("## Named Entity + Role Classification\n")
        if ner_cfg:
            f.write(f"- **NER Model:** {ner_cfg.get('spacy_model', 'unknown')}\n")
            f.write(
                f"- **Role Classifier:** {ner_cfg.get('role_classifier_model', 'unknown')}\n"
            )
            f.write(f"- **Roles:** {', '.join(ner_cfg.get('role_labels', []))}\n")
        f.write("- **Report:** [`ner_roles_report.txt`](ner_roles_report.txt)\n")
        f.write(
            "- **Confusion Matrix:** ![NER Confusion Matrix](ner_roles_confusion_matrix.png)\n\n"
        )

        # Summariser Report
        f.write("## Summarisation: April Events\n")
        if summary_cfg:
            f.write(f"- **Model Used:** {summary_cfg.get('model', 'unknown')}\n")
        f.write("- **Stats:** [`summariser_stats.txt`](summariser_stats.txt)\n")
        f.write("- **Plot:** ![Length Plot](summary_length_plot.png)\n\n")

        # Output Files
        f.write("## Output Files\n")
        f.write("| File | Description |\n")
        f.write("|------|-------------|\n")
        if os.path.exists(subcat_csv):
            f.write("| bbc_subcategories.csv | Sub-category predictions |\n")
        f.write("| ner_roles.csv | NER + Role predictions |\n")
        f.write("| april_summaries.csv | Summarised April articles |\n")
        f.write("| pipeline_timing.csv | Runtime breakdown |\n")

    print(f" Markdown report generated: {report_md}")

    # Write timing summary
    timing_csv = os.path.join(output_dir, "pipeline_timing.csv")
    with open(timing_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Step", "Seconds"])
        for step, duration in timings.items():
            writer.writerow([step, f"{duration:.2f}"])
        writer.writerow(["Total", f"{sum(timings.values()):.2f}"])
        writer.writerow(["Classifier Model", model_name or "N/A"])
        writer.writerow(["Sample Size", str(sample_size)])

    print(f" Timing saved to: {timing_csv}")
