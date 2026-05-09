import os
import yaml
from typing import Optional

from dotenv import load_dotenv

from src.registries.metrics_registry import METRICS, METRIC_FACTORIES


load_dotenv(override=True)
SUMMARY_VER = os.getenv("SUMMARY_VER")
EVALUATION_CONFIG_PATH = os.getenv("EVALUATION_CONFIG_PATH")
FILE_EXTENSION = os.getenv("FILE_EXTENSION")


# TODO: Load cpu or gpu metrics based on the CLI command
def load_metrics(
    lang: str,
    cfg_path: str = os.getenv("EVALUATION_CONFIG_PATH", "src/conf/config.yaml"),
) -> dict:
    try:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
            cfg_metrics = cfg["metrics"]
    except Exception as e:
        raise ValueError(f"Error loading metrics config from {cfg_path}: {e}")

    metrics = {}
    multilingual = lang != "en"

    for name, enabled in cfg_metrics.items():
        # Return metrics that are enabled
        if not enabled:
            continue  # skip disabled metrics

        properties = METRICS[name]

        # and match the language
        if properties.multilingual == multilingual or lang == "en":
            factory = METRIC_FACTORIES.get(name)
            metrics[name] = factory(lang)
    return metrics


def load_checkpoint(file_path):
    """Create checkpoint file if missing and return list of the already evaluated docs."""
    if not os.path.exists(file_path):
        # Create empty checkpoint file
        with open(file_path, "w", encoding="utf-8"):
            return []
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().splitlines()


def get_candidate_filenames(source_doc, candidates_dir, gold_dir=Optional[str], randoms=False):
    # Actual candidate summaries
    candidate_summaries = [doc for doc in os.listdir(candidates_dir) if doc.startswith(f"{source_doc}_")]

    if randoms:
        # 10 randoms
        other_summaries = [
            doc
            for doc in os.listdir(gold_dir)
            if not doc.startswith(f"{source_doc}_") and doc.endswith(f"{SUMMARY_VER}{FILE_EXTENSION}")
        ]
        candidate_summaries.extend(other_summaries[:10])

    # Source summary
    candidate_summaries.insert(0, source_doc)

    return candidate_summaries


def get_candidate_metadata(candidate_file, source_doc, gold_dir, candidates_dir):
    # Source
    if candidate_file == source_doc:
        candidate_path = os.path.join(gold_dir, f"{candidate_file}{SUMMARY_VER}{FILE_EXTENSION}")
        candidate_variant = "source"
    # Other (gold) summaries
    elif candidate_file.endswith(f"{SUMMARY_VER}{FILE_EXTENSION}"):
        candidate_path = os.path.join(gold_dir, candidate_file)
        candidate_variant = candidate_file.removesuffix(f"{FILE_EXTENSION}")
    # Candidate / Destroyed summaries
    else:
        candidate_path = os.path.join(candidates_dir, candidate_file)
        candidate_variant = candidate_file.removeprefix(f"{source_doc}_").removesuffix(f"{FILE_EXTENSION}")

    return candidate_path, candidate_variant


def load_source_texts(source_docs: list, source_dir: str):
    texts = []
    for source_doc in source_docs:
        with open(os.path.join(source_dir, source_doc), "r", encoding="utf-8") as f:
            texts.append(f.read())
    return texts


def load_candidate_texts(source_doc: str, candidate_files: list, gold_dir: str, candidates_dir: str):
    texts = []
    for candidate_file in candidate_files:
        candidate_path, candidate_variant = get_candidate_metadata(candidate_file, source_doc, gold_dir, candidates_dir)
        with open(candidate_path, "r", encoding="utf-8") as f:
            texts.append((candidate_variant, f.read()))
    return texts


def append_score(data, source_file, type, method, candidate_variant, result, duration):
    """Append the evaluation score and metadata to the evaluation results dataset.

    Args:
        data (dict): The evaluation results dataset.
        source_file (str): The file name of the source document.
        type (str): The type of evaluation (e.g., "N-gram", "Probabilistic").
        method (str): The method used for evaluation (e.g., "Rouge1", "Rouge2").
        candidate_variant (str): The variant of the candidate summary being evaluated (e.g. "randomly_swapped_words", "inserted_sentence").
        result (float): The evaluation score.
        duration (float): The duration taken for the evaluation.

    """
    data["source_doc"].append(source_file)
    data["eval_type"].append(type)
    data["eval_method"].append(method)
    data["variant"].append(candidate_variant)
    data["score"].append(result)
    data["duration"].append(duration)
