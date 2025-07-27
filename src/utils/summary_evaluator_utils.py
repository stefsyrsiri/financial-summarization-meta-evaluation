import os

from dotenv import load_dotenv

load_dotenv(override=True)
SUMMARY_VER = os.getenv("SUMMARY_VER")
FILE_EXTENSION = os.getenv("FILE_EXTENSION")


def get_candidate_filenames(instance, source_doc):
    # Actual candidate summaries
    candidate_summaries = [doc for doc in os.listdir(instance.candidates_dir) if doc.startswith(f"{source_doc}_")]

    # 10 randoms
    other_summaries = [doc for doc in os.listdir(instance.gold_dir) if not doc.startswith(f"{source_doc}_") and doc.endswith(f"{SUMMARY_VER}{FILE_EXTENSION}")]
    candidate_summaries.extend(other_summaries[:10])

    # Source summary
    candidate_summaries.insert(0, source_doc)

    return candidate_summaries


def get_candidate_metadata(instance, candidate_file, source_doc):
    # Source
    if candidate_file == source_doc:
        candidate_path = os.path.join(instance.gold_dir, f"{candidate_file}{SUMMARY_VER}{FILE_EXTENSION}")
        candidate_variant = "source"
    # Other (gold) summaries
    elif candidate_file.endswith(f"{SUMMARY_VER}{FILE_EXTENSION}"):
        candidate_path = os.path.join(instance.gold_dir, candidate_file)
        candidate_variant = candidate_file.removesuffix(f"{FILE_EXTENSION}")
    # Candidate / Destroyed summaries
    else:
        candidate_path = os.path.join(instance.candidates_dir, candidate_file)
        candidate_variant = candidate_file.removeprefix(f"{source_doc}_").removesuffix(f"{FILE_EXTENSION}")

    return candidate_path, candidate_variant


def load_candidate_texts(instance, source_doc: str, candidate_files: list):
    texts = []
    for candidate_file in candidate_files:
        candidate_path, candidate_variant = get_candidate_metadata(instance, candidate_file, source_doc)
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
