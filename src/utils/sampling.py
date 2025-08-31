import os
import json
import random
from loguru import logger

def get_sample_docs(
        source_docs:list,
        n_samples:int,
        sample_docs_path:str = "results/sampling.txt",
        seeds_path:str = "results/seeds.json"
        ) -> list:

    """Get a list of sampled documents.
    The documents from the first n samples are saved to be used later.
    The first time random seeds are generated and used for the sampling.
    These seeds are also saved for reproducibility.

    Args:
        source_docs (list): The list of source documents to sample from.
        n_samples (int): The number of samples to draw.
        sample_docs_path (str): The file path of the sampled documents.
        seeds_path (str): The file path of the random seeds.

    Returns:
        list: A list of sampled documents.
    """
    # Get existing samples
    sampling_file = sample_docs_path
    if os.path.exists(sampling_file):
        with open(sampling_file, "r") as f:
            sampled_docs = list(set(f.read().splitlines()))  # Distinct docs to avoid duplicates
            logger.info(f"Found {len(sampled_docs)} sampled documents in {sampling_file}.")
            return sampled_docs
    # Sample documents if not already sampled
    else:

        # Get seeds
        if os.path.exists(seeds_path):
            seeds = json.load(open(seeds_path, "r"))
        else:
            seeds = [random.randint(0, 2**32 - 1) for _ in range(n_samples)]  # Max seed value taken from 'class numpy.random.RandomState(seed=None)' documentation
            with open(seeds_path, "w") as f:
                json.dump(seeds, f)

        # Get docs
        sampled_docs = []
        for seed in seeds:
            random.seed(seed)
            sampled_docs.extend(random.sample(source_docs, 182))  # sampling without replacement
        with open(sampling_file, "a") as f:
            for doc in sampled_docs:
                    f.write(f"{doc}\n")
        return list(set(sampled_docs))  # Distinct docs to avoid duplicates
