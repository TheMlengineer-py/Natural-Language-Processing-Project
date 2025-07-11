#  HMLR 2025 NLP Pipeline (Agent-Based)

This project solves the **HMLR Data Scientist Challenge (NLP Task 1)** using modern LLMs, spaCy, and transformers with an agent-style modular design.

---

##  Features

-  BBC dataset ingestion and cleaning
-  Sub-category classification using zero-shot LLMs (BART, T5)
-  Named Entity Recognition + Role classification via LLM reasoning
-  April event summarisation using BART
-  Evaluation: F1 Score, Confusion Matrix
-  Model comparison between BART and T5
- Agent-friendly tools (LangChain-compatible)

---

## how to run

```bash
# Clone & install
git clone https://github.com/your-username/Natural-Language-Processing-Project.git
cd Natural-Language-Processing-Project
pip install -r requirements.txt

# Run the full pipeline
python main.py

