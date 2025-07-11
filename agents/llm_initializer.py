"""
llm_initializer.py

Central hub to load HuggingFace LLMs as inference pipelines.
Used by all module: classification, summarisation, role reasoning.
"""

from transformers import pipeline


def load_zero_shot_model(model_name="facebook/bart-large-mnli"):
    """
    Load a HuggingFace zero-shot classification pipeline.

    Args:
        model_name (str): Model checkpoint from HuggingFace

    Returns:
        transformers.Pipeline: Configured zero-shot classification pipeline
    """
    print(f" Loading Zero-Shot Model: {model_name}")
    return pipeline("zero-shot-classification", model=model_name)


def load_summarizer(model_name="facebook/bart-large-cnn"):
    """
    Load a summarisation model pipeline.

    Args:
        model_name (str): HuggingFace summarisation model ID

    Returns:
        transformers.Pipeline: Summariser pipeline
    """
    print(f" Loading Summarizer Model: {model_name}")
    return pipeline("summarization", model=model_name)
