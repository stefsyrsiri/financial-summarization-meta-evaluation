import os
import re
from time import perf_counter

import numpy as np
from loguru import logger
from tqdm import tqdm

from modules.summary_corruptor import SummaryCorruptor
from modules.summary_generator import SummaryGenerator
from src.utils.summary_evaluator_utils import get_candidate_metadata, get_candidate_filenames


def generate_noisy_summaries(
    source_docs, gold_summaries_dir, candidate_summaries_dir, summary_ver, file_extension, truncate_for_bert
):
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

    summary_generator = SummaryGenerator(
        source_docs=source_docs,
        gold_dir=gold_summaries_dir,
        candidate_dir=candidate_summaries_dir,
        truncate_for_bert=truncate_for_bert,
    )

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
            corruptor = SummaryCorruptor(
                input_summary=gold_summary, noise_percentage=0.9, truncate_for_bert=truncate_for_bert
            )

            # Summary Generation
            percentages = np.round(np.linspace(0.9, 0.1, num=5), 1).tolist()
            for percentage in percentages:
                corruptor.noise_percentage = percentage
                summary_generator.generate_noisy_summaries(doc_id=doc, corruptor=corruptor, noise_percentage=percentage)
        except Exception as e:
            logger.exception(f"Failed processing for gold summary {doc}: {e}")


def generate_gold_summaries(
    source_docs,
    source_dir,
    candidate_summaries_dir,
    gold_summaries_dir,
    extracted_summaries_dir,
    file_extension,
    extractor,
):
    logger.info("Generating gold summaries from source docs.")

    # Every source doc
    for doc in tqdm(source_docs, desc="Processing documents"):
        logger.info(f"Processing annual report '{doc}' for summary extraction.")
        source_doc_path = os.path.join(source_dir, f"{doc}{file_extension}")

        if not os.path.exists(source_doc_path):
            logger.warning(f"File not found for {doc}, skipping.")
            continue
        with open(source_doc_path, mode="r", encoding="utf-8") as source_f:
            source_doc = source_f.read()

        # Every candidate summary
        candidate_summaries = get_candidate_filenames(source_doc=doc, candidates_dir=candidate_summaries_dir)

        for candidate_file in candidate_summaries:
            candidate_path, candidate_variant = get_candidate_metadata(
                candidate_file=candidate_file,
                source_doc=doc,
                gold_dir=gold_summaries_dir,
                candidates_dir=candidate_summaries_dir,
            )
            if (
                candidate_variant == "source"
                or re.findall(r"inserted_sentence_0.[159]", candidate_variant)
                or re.findall(r"repeated_sentence_0.[159]", candidate_variant)
            ):
                try:
                    logger.info(f"Extracting summary for candidate summary: {candidate_file}")
                    with open(candidate_path, mode="r", encoding="utf-8") as cand_f:
                        candidate_summary = cand_f.read()
                except FileNotFoundError as e:
                    logger.exception(f"File not found: {e}. Skipping candidate_file: {candidate_file}.")
                    continue

                try:
                    start_time = perf_counter()
                    ref, _ = extractor.extract_reference_summary(source_doc, candidate_summary)
                    duration = perf_counter() - start_time

                    if not candidate_file.endswith(".txt"):
                        candidate_file = candidate_file + ".txt"
                    with open(
                        os.path.join(extracted_summaries_dir, f"{candidate_file}"),
                        mode="w",
                        encoding="utf-8",
                    ) as gold_f:
                        gold_f.write(ref)

                    with open("results/en_ngram_duration_trunc.csv", mode="a", encoding="utf-8") as duration_f:
                        duration_f.write(f"{candidate_file.strip('.txt')},{duration}\n")

                    logger.info(f"Extracted gold summary for {candidate_file} saved to {extracted_summaries_dir}.")
                except Exception as e:
                    logger.exception(f"Failed to extract gold summary for {candidate_file}: {e}")
