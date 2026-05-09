import os
import json
import random
from typing import Optional
from loguru import logger


def get_sample_docs(
    sampled_docs_path: str,
    seeds_path: str,
    source_docs: Optional[list] = None,
    n_samples: Optional[int] = None,
    sample_k: Optional[int] = None,
) -> list:
    """Get a list of sampled documents.
    The documents from the first n samples are saved to be used later.
    The first time random seeds are generated and used for the sampling.
    These seeds are also saved for reproducibility.

    Args:
        sampled_docs_path (str): The file path of the sampled documents.
        seeds_path (str): The file path of the random seeds.
        source_docs (list): The list of source documents to sample from.
        n_samples (int): The number of samples to draw.

    Returns:
        list: A list of sampled documents.
    """
    # Get existing samples
    logger.debug(f"Sampling file path: {sampled_docs_path}")
    if os.path.exists(sampled_docs_path):
        with open(sampled_docs_path, "r") as f:
            sampled_docs = list(set(f.read().splitlines()))  # Distinct docs to avoid duplicates
            logger.info(f"Found {len(sampled_docs)} sampled documents in {sampled_docs_path}.")
            return sampled_docs
    # Sample documents if not already sampled
    else:
        # Get seeds
        if os.path.exists(seeds_path):
            seeds = json.load(open(seeds_path, "r"))
        else:
            # Max seed value taken from 'class numpy.random.RandomState(seed=None)' documentation
            seeds = [random.randint(0, 2**32 - 1) for _ in range(n_samples)]
            with open(seeds_path, "w") as f:
                json.dump(seeds, f)

        # Get docs
        sampled_docs = []
        for seed in seeds:
            random.seed(seed)
            sampled_docs.extend(random.sample(source_docs, sample_k))  # sampling without replacement
        with open(sampled_docs_path, "a") as f:
            for doc in sampled_docs:
                f.write(f"{doc}\n")
        return list(set(sampled_docs))  # Distinct docs to avoid duplicates
