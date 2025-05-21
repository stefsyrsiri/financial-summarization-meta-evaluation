import math

from loguru import logger


def get_swap_indices(instance):
    """Helper method for the random_swap_words and consecutive_swap_words methods.
        Prepares the summary indices and target number used for swapping.

    Returns:
        The summary indices and the target number used for sampling summary indices.

    """
    summary_len = len(instance.words)
    logger.debug(f"Total words: {summary_len}")

    target_number = math.ceil(summary_len * instance.noise_percentage) // 2
    logger.debug(f"Number of swaps: {target_number}")
    return target_number
