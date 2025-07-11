#  HMLR NLP Pipeline – Solution Design Document

## Objective

The aim of this project is to design and deploy a robust, modular Natural Language Processing (NLP) pipeline using the BBC News dataset to:

- Classify articles into **fine-grained subcategories** using large language models (LLMs)
- Perform **Named Entity Recognition (NER)** and classify media personalities by role
- Generate **summaries** of articles referencing **April events**
- Output detailed evaluation metrics and generate markdown + PDF reports

---

##  System Architecture

```plaintext
                ┌──────────────────────────────┐
                │  Raw BBC Dataset (UCD)       │
                └────────────┬─────────────────┘
                             │
         ┌──────────────────▼────────────────────┐
         │        Preprocessing & Cleaning       │
         └──────────────────┬────────────────────┘
                             │
                 ┌──────────▼──────────┐
                 │     main.py         │
                 └──────────┬──────────┘
                            │
 ┌──────────────────────────┼────────────────────────────────────────┐
 │                          │                                        │
▼                           ▼                                        ▼
Sub-Category         Named Entity + Role                        Summariser
 Classification        Classification                         (April Events)
(module/classifier)    (module/ner_agent)                      (module/summariser)
                            │                                        │
                            ▼                                        ▼
                    Evaluation & Reports                 Evaluation & Stats

## Tech Stack
| Layer         | Tools & Libraries                                |
| ------------- | ------------------------------------------------ |
| LLMs          | `transformers`, `facebook/bart`, `flan-t5`, etc  |
| NLP/NER       | `spaCy`, `en_core_web_trf`                       |
| Preprocessing | `nltk`, `tqdm`, `pandas`                         |
| Evaluation    | `sklearn`, `matplotlib`, `seaborn`               |
| Reporting     | Markdown, `pandoc` (optional for PDF export)     |
| CI/CD         | GitHub Actions (CI lint, test, coverage, secure) |
| Testing       | `pytest`, `pytest-cov`                           |


## Module Structure

Natural-Language-Processing-Project/
├── .github/                    # cicd
│   ├── workflows/
├── module/                    # Main business logic
│   ├── classifier_agent.py
│   ├── ner_agent.py
│   ├── summarizer_agent.py
│   ├── evaluator.py
│   ├── loader.py
│   ├── downloader.py
│   ├── config_loader.py
│   └── report_generator.py
├── config/
│   ├── agents.yaml            # All LLM & pipeline hyperparams
│   └── pipeline.yaml          # Sample size, mode, etc.
├── data/
│   ├── bbc/                   # Raw text dataset
│   └── bbc_processed/         # Cleaned + tokenized text
├── outputs/
│   ├── bbc_subcategories.csv
│   ├── ner_roles.csv
│   ├── april_summaries.csv
│   └── reports/
│       ├── classifier_report.txt
│       ├── summary_length_plot.png
│       └── report.md / report.pdf
├── tests/                     # Unit tests (pytest)
├── main.py                    # Orchestration pipeline
├── requirements.txt
├── README.md
└── run_all.sh                 # Full setup + run script


## Pipeline Configuration
 config/agents.yaml
Stores all model names and hyperparameters for:

Sub-category classifier

Role classifier (NER)

Summariser model

config/pipeline.yaml
Controls:

Sample size (e.g. 200 for CPU)

Step skipping

Output/report locations

## Testing & CI/CD

Tests
Unit tests for each agent

Fixtures for fake articles

Coverage: pytest-cov

## GitHub Actions
Runs on PR to main, dev, feature/**

Steps:

Lint: flake8, black, ruff

Tests: pytest

Security: bandit

Static analysis: pylint

Coverage report

## Evaluation Metrics
Each agent generates:

classification_report.txt for label accuracy

Confusion matrix PNGs for multi-class inspection

Summary plot statistics (avg/median length)

Timing breakdown of all stages

Markdown → PDF report for final submission

Key Design Principles
Modular: each task isolated in its own agent

Configurable: pipeline behavior controlled via YAML

Testable: CI integrated + mocked pipelines

Reproducible: sample size, models, and preprocessing reproducible

Explainable: outputs + stats logged clearly in outputs/reports/

## Execution Instructions

'''bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download + preprocess data
python main.py

# 3. Run tests
pytest tests/ --cov=module

# 4. Run all steps end-to-end
bash run_all.sh
