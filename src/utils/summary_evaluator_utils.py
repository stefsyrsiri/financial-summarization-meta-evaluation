
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
