import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_scores(
    dfs: list[pd.DataFrame],
    title: str,
    noise_variant: str,
    languages: list[str] = ["English", "Greek", "Spanish"],
    hue: str = "eval_method",
    save_and_close: bool = False,
    file_name: str = None,
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
    if save_and_close:
        fig, axes = plt.subplots(3, 1, figsize=(8, 15), sharex=True)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for ax, df, lang in zip(axes, dfs, languages):
        # Base lineplot for source and noise variant
        sns.lineplot(
            data=df[(df["variant_type"] == noise_variant) | (df["variant_type"] == "source")],
            x="noise_percentage",
            y="score",
            hue=hue,
            linewidth=2,
            estimator=np.mean,
            errorbar="sd",
            ax=ax,
        )

        # Average for 'random_summary'
        random_summary_mean = df[df["variant_type"] == "random_summary"]["score"].mean()
        ax.axhline(
            y=random_summary_mean,
            linestyle="--",
            color="gray",
            label="AVG Random Summary Score",
        )

        ax.set_title(f"{lang}")
        ax.set_xlabel("Noise Percentage")
        ax.set_ylabel("AVG Evaluation Score")
        ax.set_ylim(0, 1.05)
        ax.set_yticks(np.arange(0.0, 1.1, step=0.1))
        ax.legend()

    fig.suptitle(title, fontsize=16, y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_and_close:
        plt.savefig(f"../figures/{file_name if file_name else noise_variant}_lineplot.png", dpi=300)
        plt.close(fig)
    else:
        plt.show()


def plot_scores_dist(df, noise_variant, title, x="language", y="score", hue="eval_method"):
    plt.figure(figsize=(15, 7))
    sns.boxplot(data=df[df["variant_type"] == noise_variant], x=x, y=y, hue=hue)
    plt.title(title)
    plt.ylabel(y)
    plt.show()


def t_corr(df, lang, noise_variant, lang_specific=False):
    vmin = df[["pearson", "spearman", "kendall"]].min().min()
    vmax = df[["pearson", "spearman", "kendall"]].max().max()

    if lang_specific:
        method_filter = df["eval_method"].isin(["BARTScore", "BLEURT"])
    else:
        method_filter = ~df["eval_method"].isin(["BARTScore", "BLEURT"])

    return (
        df[(df["language"] == lang) & (df["variant_type"] == noise_variant) & method_filter]
        .drop(columns=["language", "variant_type"])
        .style.set_caption(lang.title() + " - " + noise_variant)
        .background_gradient(cmap="coolwarm_r", vmin=vmin, vmax=vmax)
        .hide(axis="index")
    )


def t_corr_all(df, noise_variant, index_cols=["eval_type", "eval_method"]):
    # English
    en = (
        df.loc[(df["variant_type"] == noise_variant) & (df["language"] == "English")]
        .iloc[:, 2:]
        .rename(columns={"spearman": "English", "p_value": "English_p"})
    )
    # Greek
    el = (
        df.loc[(df["variant_type"] == noise_variant) & (df["language"] == "Greek")]
        .iloc[:, 2:]
        .rename(columns={"spearman": "Greek", "p_value": "Greek_p"})
    )
    # Spanish
    es = (
        df.loc[(df["variant_type"] == noise_variant) & (df["language"] == "Spanish")]
        .iloc[:, 2:]
        .rename(columns={"spearman": "Spanish", "p_value": "Spanish_p"})
    )
    merged = en.merge(el, on=index_cols, how="left").merge(es, on=index_cols, how="left")
    return merged.rename(columns={"eval_type": "evaluation type", "eval_method": "evaluation method"})


def t_corr_all_formatted(
    df,
    noise_variant,
    cmap="coolwarm_r",
    subset=["English", "Greek", "Spanish"],
    index_cols=["eval_type", "eval_method"],
):
    return (
        t_corr_all(df=df, noise_variant=noise_variant, index_cols=index_cols)
        .style.background_gradient(cmap=cmap, subset=subset)
        .format({col: "{:.2f}" for col in subset + ["English_p", "Greek_p", "Spanish_p"]})
    )


def save_table(df, noise_variant, file_name=None):
    t_corr_all(df=df, noise_variant=noise_variant).to_latex(
        buf=f"../tables/{file_name if file_name else noise_variant}.tex", na_rep="N/A", float_format="%.2f"
    )
