"""
summarizer_agent.py

 Summarizes April-related events using a transformer model.

- Filters articles that mention "April"
- Applies summarization LLM (e.g. BART, T5)
- Evaluates summary quality via length reduction
- Configurable via YAML
"""

from transformers import pipeline
import pandas as pd
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
import os


def summarise_april_events(
    df: pd.DataFrame,
    model_name: str,
    max_length: int = 80,
    min_length: int = 30,
    do_sample: bool = False,
    output_dir: str = "outputs",
) -> pd.DataFrame:
    """
    Summarizes articles mentioning April using transformer model.

    Args:
        df (pd.DataFrame): Full article dataset
        model_name (str): HuggingFace summarization model ID
        max_length (int): Max output tokens
        min_length (int): Min output tokens
        do_sample (bool): Sampling mode for generation
        output_dir (str): Folder to store reports

    Returns:
        pd.DataFrame: DataFrame with summaries and source text
    """
    print(f" Loading summarizer model: {model_name}")
    summarizer = pipeline("summarization", model=model_name)

    april_df = df[
        df["text"].str.contains(r"\bApril\b", flags=re.IGNORECASE, regex=True)
    ].copy()
    results = []

    for _, row in tqdm(april_df.iterrows(), total=len(april_df)):
        original = row["text"][:1000]
        try:
            prompt = f"Summarise the following text, focusing only on events happening in April:\n{original}"
            summary = summarizer(
                prompt,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
            )[0]["summary_text"]
        except Exception as e:
            summary = "Summarisation failed."

        results.append(
            {
                "source_text": original[:300],
                "summary": summary,
                "orig_len": len(original.split()),
                "summary_len": len(summary.split()),
            }
        )

    summary_df = pd.DataFrame(results)

    # Save summarisation results
    csv_path = os.path.join(output_dir, "april_summaries.csv")
    txt_path = os.path.join(output_dir, "summariser_stats.txt")

    summary_df.to_csv(csv_path, index=False)

    #  Plot summary length comparison
    try:
        plt.figure(figsize=(8, 5))
        plt.scatter(
            summary_df["orig_len"], summary_df["summary_len"], alpha=0.6, color="teal"
        )
        plt.plot(
            [0, max(summary_df["orig_len"])],
            [0, max(summary_df["orig_len"])],
            linestyle="--",
            color="gray",
        )
        plt.xlabel("Original Length (words)")
        plt.ylabel("Summary Length (words)")
        plt.title("Original vs Summary Length (April Events)")
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "reports/summary_length_plot.png")
        plt.savefig(plot_path)
        print(f" Summary length plot saved to: {plot_path}")
    except Exception as e:
        print(f" Plotting skipped: {e}")

    try:
        reduction = (
            1 - (summary_df["summary_len"].mean() / summary_df["orig_len"].mean())
        ) * 100
        with open(txt_path, "w") as f:
            f.write(f" Summary Evaluation Report\n")
            f.write(f"Articles Mentioning April: {len(summary_df)}\n")
            f.write(
                f"Average Original Length: {summary_df['orig_len'].mean():.2f} words\n"
            )
            f.write(
                f"Average Summary Length: {summary_df['summary_len'].mean():.2f} words\n"
            )
            f.write(f"Avg Length Reduction: {reduction:.2f}%\n")
        print(f" Summary statistics saved to: {txt_path}")
    except Exception as e:
        print(f" Summary evaluation skipped: {e}")

    return summary_df


# --------version 1---#
# """
# summarizer_agent.py
#
# This module filters articles that mention "April" and uses a transformer-based
# summarisation model (like BART) to create short summaries focused on those events.
# """
#
# from transformers import pipeline
# import pandas as pd
# import re
# from tqdm import tqdm
#
# def summarise_april_events(df: pd.DataFrame, model_name="facebook/bart-large-cnn") -> pd.DataFrame:
#     """
#     Filters articles with April references and generates event summaries using LLM.
#
#     Args:
#         df (pd.DataFrame): Articles with `text` column
#         model_name (str): HuggingFace summarisation model (default is BART)
#
#     Returns:
#         pd.DataFrame: Rows with original snippet and April-focused summary
#     """
#     print(f" Loading summariser model: {model_name}")
#     summarizer = pipeline("summarization", model=model_name)
#
#     # Filter for April references (case-insensitive)
#     april_df = df[df['text'].str.contains(r'\bApril\b', flags=re.IGNORECASE, regex=True)]
#
#     results = []
#     for _, row in tqdm(april_df.iterrows(), total=len(april_df)):
#         try:
#             # Create a focused prompt for summarisation
#             prompt = f"Summarise the following text, focusing only on events happening in April:\n{row['text'][:1000]}"
#             summary = summarizer(prompt, max_length=80, min_length=30, do_sample=False)[0]['summary_text']
#         except Exception as e:
#             summary = "Summarisation failed."
#
#         results.append({
#             "source_text": row['text'][:300],
#             "summary": summary
#         })
#
#     return pd.DataFrame(results)
