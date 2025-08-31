import os

import numpy as np
from loguru import logger
from tqdm import tqdm

from modules.summary_corruptor import SummaryCorruptor
from modules.summary_generator import SummaryGenerator


def generate_noisy_summaries(source_docs, gold_summaries_dir, candidate_summaries_dir, summary_ver, file_extension, truncate_for_bert):
    """Generate noisy summaries from the gold summaries.

    Args:
        source_docs (list): List of documents to process.
        gold_summaries_dir (str): Directory containing the gold summaries.
        candidate_summaries_dir (str): Directory to save the generated summaries.
        summary_ver (str): Version of the summary files.
        file_extension (str): File extension for the summary files.
    """
    logger.info("Generating candidate summaries from gold summaries.")

    # Noisy summaries
    if not os.path.isdir(candidate_summaries_dir):
        os.makedirs(candidate_summaries_dir, exist_ok=True)

    summary_generator = SummaryGenerator(source_docs=source_docs, gold_dir=gold_summaries_dir, candidate_dir=candidate_summaries_dir, truncate_for_bert=truncate_for_bert)

    for doc in tqdm(source_docs, desc="Processing documents"):
        logger.info(f"Processing annual report '{doc}' for noisy summary generation.")
        gold_summary_path = os.path.join(gold_summaries_dir, f"{doc}{summary_ver}{file_extension}")
        if not os.path.exists(gold_summary_path):
            logger.warning(f"File not found for {doc}, skipping.")
            continue
        with open(file=gold_summary_path, mode="r", encoding="utf-8") as file:
            gold_summary = file.read()
            if gold_summary.strip() == "":
                continue

        # Summary Destruction
        try:
            corruptor = SummaryCorruptor(input_summary=gold_summary, noise_percentage=0.9, truncate_for_bert=truncate_for_bert)

            # Summary Generation
            percentages = np.round(np.linspace(0.9, 0.1, num=5), 1).tolist()
            for percentage in percentages:
                corruptor.noise_percentage = percentage
                summary_generator.generate_noisy_summaries(doc_id=doc, corruptor=corruptor, noise_percentage=percentage)
        except Exception as e:
            logger.exception(f"Failed processing for gold summary {doc}: {e}")
