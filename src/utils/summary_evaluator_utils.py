
def append_score(instance, source_file, type, method, candidate_variant, result, duration):
    """Append the evaluation score and metadata to the evaluation results dataset.

    Args:
        instance (SummaryEvaluator): The instance of the SummaryEvaluator class.
        source_file (str): The file name of the source document.
        type (str): The type of evaluation (e.g., "N-gram", "Probabilistic").
        method (str): The method used for evaluation (e.g., "Rouge1", "Rouge2").
        candidate_variant (str): The variant of the candidate summary being evaluated (e.g. "randomly_swapped_words", "inserted_sentence").
        result (float): The evaluation score.
        duration (float): The duration taken for the evaluation.

    """
    instance.data["source_doc"].append(source_file)
    instance.data["eval_type"].append(type)
    instance.data["eval_method"].append(method)
    instance.data["variant"].append(candidate_variant)
    instance.data["score"].append(result)
    instance.data["duration"].append(duration)
