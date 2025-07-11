"""
tools.py

This module defines callable tools (functions) that can be plugged into an agent.
Each tool is a wrapper around LLM logic for use in LangChain, CLI, or interactive notebooks.
"""

from transformers import pipeline


def role_classifier_tool(text: str) -> str:
    """
    Tool: Classify someone's profession from a sentence.

    Args:
        text (str): A sentence mentioning a person

    Returns:
        str: Predicted role
    """
    model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    labels = [
        "Politician",
        "Musician",
        "TV Personality",
        "Film Actor",
        "Author",
        "Business Executive",
    ]
    result = model(text, candidate_labels=labels)
    return result["labels"][0]


def summarise_tool(text: str) -> str:
    """
    Tool: Summarise April-related events from a given article.

    Args:
        text (str): Article content

    Returns:
        str: Short summary
    """
    prompt = (
        f"Summarise the following article, focusing only on events in April:\n{text}"
    )
    model = pipeline("summarization", model="facebook/bart-large-cnn")
    return model(prompt[:1024])[0]["summary_text"]
