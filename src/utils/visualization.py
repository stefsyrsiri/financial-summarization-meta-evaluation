from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display, HTML


def plot_scores(
        dfs: List[pd.DataFrame],
        title: str,
        noise_variant: str,
        languages: List[str] = ["English", "Greek", "Spanish"],
        ):
    """
    Plot the scores for different languages and noise percentages.

    Args:
        dfs (list): List of DataFrames containing the scores.
        title (str): Title of the plot.
        noise_variant (str): The type of noise.
        languages (list): List of language names corresponding to the DataFrames.
    """
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for ax, df, lang in zip(axes, dfs, languages):

        # Base lineplot for source and noise variant
        sns.lineplot(
            data=df[(df["variant_type"] == noise_variant) | (df["variant_type"] == "source")],
            x="noise_percentage",
            y="score",
            hue="eval_method",
            linewidth=2,
            estimator=np.mean,
            errorbar="sd",
            ax=ax
        )

        # Average for 'random_summary'
        random_summary_mean = df[df["variant_type"] == "random_summary"]["score"].mean()
        ax.axhline(
            y=random_summary_mean,
            linestyle="--",
            color="gray",
            label="AVG Randomness"
        )

        ax.set_title(f"{lang}")
        ax.set_xlabel("Noise Percentage")
        ax.set_ylabel("Average Score" if lang == "English" else "")
        ax.set_ylim(0, 1.05)
        ax.set_yticks(np.arange(0.0, 1.1, step=0.1))
        ax.legend()

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def t_corr(df, lang, noise_variant, lang_specific=False):
    vmin = df[['pearson', 'spearman', 'kendall']].min().min()
    vmax = df[['pearson', 'spearman', 'kendall']].max().max()

    if lang_specific:
        method_filter = df['eval_method'].isin(['BARTScore', 'Bleurt'])
    else:
        method_filter = ~df['eval_method'].isin(['BARTScore', 'Bleurt'])

    return df[
        (df['language'] == lang)
        & (df['variant_type'] == noise_variant)
        & method_filter
        ].drop(columns=['language', 'variant_type']).style.set_caption(lang.title()+' - '+noise_variant).background_gradient(cmap='coolwarm_r', vmin=vmin, vmax=vmax).hide(axis='index')


def t_corr_all(df, noise_variant):

    styled = [t_corr(df, lang=lang, noise_variant=noise_variant) for lang in ['English', 'Greek', 'Spanish']]

    html = '<div style="display: flex;">'
    for s in styled:
        html += f'<div>{s.to_html()}</div>'
    html += '</div>'

    display(HTML(html))
