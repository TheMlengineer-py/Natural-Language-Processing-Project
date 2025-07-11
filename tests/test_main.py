"""
test_main.py

Unit test for `main.py` pipeline execution.
Automatically adjusts PYTHONPATH to import from project root.
"""

import os
import sys
import pytest
import logging
import warnings
from unittest import mock

# Suppress external deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="typer")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="spacy")

# Add project root to PYTHONPATH
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from main import run_pipeline


@pytest.fixture(scope="module")
def test_output_dirs():
    # Use test-specific output folders
    test_outputs = os.path.join(ROOT_DIR, "test_outputs")
    test_reports = os.path.join(test_outputs, "reports")

    os.makedirs(test_reports, exist_ok=True)

    # Patch environment variables or constants if needed
    with mock.patch("main.OUTPUT_DIR", test_outputs), mock.patch(
        "main.REPORTS_DIR", test_reports
    ):

        yield test_outputs, test_reports

    # Cleanup after tests
    if os.path.exists(test_outputs):
        for root, dirs, files in os.walk(test_outputs, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(test_outputs)


@mock.patch("main.load_config")
@mock.patch("main.classify_subcategories")
@mock.patch("main.extract_named_entities_and_roles")
@mock.patch("main.summarise_april_events")
def test_run_pipeline_isolated_outputs(
    mock_summarise, mock_ner, mock_classifier, mock_config, test_output_dirs, caplog
):
    import pandas as pd

    caplog.set_level(logging.INFO)

    mock_config.return_value = {
        "pipeline": {"sample_size": 2},
        "classifier": {"model": "mock-model", "top_k": 1, "max_length": 1000},
        "ner": {
            "spacy_model": "en_core_web_sm",
            "role_classifier_model": "mock-role-model",
            "role_labels": ["Film Actor"],
        },
        "summarizer": {
            "model": "mock-summary-model",
            "max_length": 100,
            "min_length": 20,
            "do_sample": False,
        },
    }

    classifier_df = pd.DataFrame(
        {
            "category": ["business", "sport"],
            "text": ["Company merger", "Cricket match"],
            "sub_category": ["mergers", "cricket"],
        }
    )
    mock_classifier.return_value = classifier_df

    ner_df = pd.DataFrame([{"name": "Tom Holland", "role": "Film Actor"}])
    mock_ner.return_value = ner_df

    summary_df = pd.DataFrame(
        [{"source_text": "An April event...", "summary": "April event summary"}]
    )
    mock_summarise.return_value = summary_df

    # Run the pipeline with patched output dirs
    run_pipeline()

    # Log check
    expected_logs = [
        "HMLR NLP pipeline started",
        "Loading BBC dataset...",
        "Using 2 articles.",
        "Using classifier model: mock-model",
        "Summarisation completed and report generated.",
    ]
    for expected in expected_logs:
        assert any(
            expected in record.message for record in caplog.records
        ), f"Missing log: '{expected}'"

    output_dir, reports_dir = test_output_dirs

    assert os.path.exists(os.path.join(output_dir, "bbc_subcategories.csv"))
    assert os.path.exists(os.path.join(output_dir, "ner_roles.csv"))
    assert os.path.exists(os.path.join(output_dir, "april_summaries.csv"))
    assert os.path.exists(os.path.join(reports_dir, "report.md"))
    assert os.path.exists(os.path.join(output_dir, "pipeline_timing.csv"))
