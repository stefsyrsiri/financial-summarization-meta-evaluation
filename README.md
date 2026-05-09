# Financial Narrative Summarization Evaluation

[![python](https://img.shields.io/badge/Python-3.10-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
<!-- ![Status](https://img.shields.io/badge/status-in--progress-yellow) -->

## 📄 Overview

In this project we evaluate evaluation methods for automatic summarization on long financial documents from the Financial Narrative Summarization (FNS) task 2023.

## Table of Contents

- [Overview](#-overview)
- [Features](#features)
- [Project structure](#️-project-structure)
- [Requirements](#requirements)
- [Installation](#️-installation)
- [Running the scripts](#-running-the-scripts)

## Features

### Data collection

This project uses 3 datasets of annual reports and their (gold) summaries.

- English
- Greek
- Spanish

Due to access limitations a script has been added to download the Greek dataset that is publicly available.

To get the Greek dataset [run the data collection script](#-running-the-scripts).

### Dataset and text statistics

You can get the statistics for all the datasets (original datasets and generated candidate summaries) and their texts by running the [stats scripts](#-running-the-scripts).

**Text statistics extracted**:

- spaCy token count
- spaCy sentence count
- BERT token count

### Generation of noisy candidate summaries

To create noisy summaries from existing ones [run the summary corruption script](#-running-the-scripts).

### Summary evaluation

To evaluate your summaries [run the summary evaluation script](#-running-the-scripts).

## 🗂️ Project structure

```sh
thesis/
│── data/                         # Source documents, gold summaries, candidate summaries
│── evaluation_methods/           # Evaluation methods 
│── notebooks/                    # Analyses of the results
│── results/                      # Evaluation results
│── samples/                      # Example untracked files
│── src/
|   ├── conf/                     # Evaluation config
|   ├── modules/
│       ├── data_collector.py     # Data collection
│       ├── stats_extractor.py    # Spacy and BERT token and sentence counts
│       ├── summary_corruptor.py  # Noise insertion
│       ├── summary_evaluator.py  # Evaluator
│       ├── summary_generator.py  # Candidate summaries generator
│       ├── tokenizer.py          # Tokenization handling
|   ├── pipelines/
│       ├── generate.py           # Candidate summaries and proxy reference summary generation pipeline
│       ├── evaluate.py           # Evaluation pipeline
|   ├── registries/               # Language and metric registries
|   ├── utils/                    # Helper functions for the main modules
├── main.py
│── README.md
```

### Requirements

#### `uv` package manager

This project uses `uv` for its dependencies. Follow the official [installation instructions](https://docs.astral.sh/uv/getting-started/installation/) depending on your OS.

#### Datasets

A `data` directory at the root is needed.

If you don't have a dataset available, you can [run the data collection script](#-running-the-scripts) to get a dataset of Greek annual reports and their gold summaries

#### Untracked files and directories

1. Create a `.env` file at the root based on the `samples/sample.env` file.
2. Create a `conf/config.yaml` file in the `src/` directory.

To generate and/or evaluate candidate summaries, configure the language variables in the `.env` file:

- `LANGUAGE`: `English`, `Greek`, `Spanish`
- `SUMMARY_VER`:
  - English: `_1`
  - Greek: `_2`
  - Spanish: `_GS1`

#### Evaluation methods

The metrics used in this project are:

| Metric type        |         Metric name        |                         How to set up                         |
|--------------------|:--------------------------:|:-------------------------------------------------------------:|
| N-gram-based       | Rouge-1, Rouge-2           | Comes with the project                                        |
| N-gram-graph-based | AutoSummENG, MeMoG, NPowER | Contact [G.Giannakopoulos](https://github.com/ggianna)        |
| Embeddings-based   | BERTScore                  | Comes with the project                                        |
| Embeddings-based   | BARTScore                  | Clone [repository](https://github.com/neulab/BARTScore)       |
| Model-based        | Bleurt                     | Clone [repository](https://github.com/google-research/bleurt) |
| Model-based        | FactCC                     | Model is used via HuggingFace                                 |
| Model-based        | LongDocFACTScore           | Comes with the project                                        |

##### Installation of BARTScore and Bleurt

Create an `evaluation_methods` directory at the root.

```bash
cd thesis
mkdir evaluation_methods
cd evaluation_methods
```

Clone the repositories inside the `evaluation_methods` directory **following the instructions of the repositories**.

```bash
git clone <the-eval-metric-repo>
```

## 🛠️ Installation

```sh
git clone https://github.com/stefsyrsiri/thesis.git
cd thesis
uv sync --locked
```

## 🚀 Running the scripts

To use the scripts, ensure that all the [requirements](#requirements) are met.

### 🔧 Run options

```sh
# Run the pipeline with selected steps:
uv run main.py [--collect] [--generate] [--evaluate] [--all]
```

#### Collect

To collect the Greek annual reports dataset run the main script with the `collect` flag:

```bash
uv run main.py --collect
```

#### Merge datasets (needed to get the dataset stats)

To merge all the datasets (all languages, all document types) and store them in a single place run the following:

```bash
uv run main.py --merge-datasets
```

#### Get statistics

To get the text statistics for the merged dataset run the following:

```bash
uv run main.py --stats
```

#### Generate

To create noisy summaries run the main script with the `generate` flag:

```bash
uv run main.py --generate
```

You can optionally use the `truncate` flag to truncate the summaries at **512 tokens** *before* applying the noise. This is useful, if you need to evaluate summaries with metrics that have token input limitations.

**Note**:
Even with truncation, it is not possible to predetermine the exact length of the final text and it is certain that it will exceed the 512 token limit and be automatically truncated. However, truncating prior to noise insertion limits the automatic truncation of the metric at inference time making the results more reliable.

#### Evaluate

To evaluate summaries run the main script with the `evaluate` flag. When running the evaluation script, you need to specify whether you're going to use CPU-bound or GPU-bound metrics by using either the `cpu` or `gpu` flag. If you want to use a reference-free metric, such as LongDocFACTScore, you need to add the `no-refs` flag.

**Note**:

LongDocFACTScore is GPU-bound as it depends on BARTScore, and therefore you need both `gpu` and `no-refs` flags.

```bash
uv run main.py --evaluate
```

#### Argument table

| Category | Argument | Description |
| :--- | :--- | :--- |
| **Data Collection** | `--collect` | Collect data. |
| **Document Stats** | `--merge-datasets` | Create a unified dataset from all the .txt files. |
| | `--stats` | Get text statistics. |
| **Sampling** | `--sample` | Sample source documents. |
| **Generation** | `--generate` | Generate noisy summaries. |
| | `--truncate` | Truncate long documents. |
| **Evaluation** | `--evaluate` | Evaluate summaries. |
| | `--cpu` | Run only CPU-bound evaluation. |
| | `--gpu` | Run only GPU-bound evaluation. |
| | `--no-refs` | Reference free evaluation (must be used with `gpu` arg). |
| | `--new` | Evaluate using extracted summaries. |
| **Extraction** | `--ngram-extract` | Run n-gram overlap-based extraction. |
| **Subset** | `--subset` | Subset of source documents to process. |
| **Workflow** | `--all` | Run all steps. |

#### Examples

```sh
# Collect (Greek) data
uv run main.py --collect

# Generate noisy summaries
uv run main.py --generate --truncate

# Evaluate summaries
uv run main.py --evaluate --gpu --subset 10

# Run all steps: collect, generate, evaluate
uv run main.py --all
```
