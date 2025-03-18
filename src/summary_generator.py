"""
This module utilizes SummaryDestructor to construct new candidate summaries.

It includes functions to check whether data already exists
and if it doesn't, it downloads it and unzips it.
"""


import os
from loguru import logger
from summary_destructor import SummaryDestructor


class SummaryGenerator:
    def __init__(
            self,
            source_docs: list,
            gold_dir: str,
            candidate_dir: str
            ):
        """Constructs candidate summaries.

        Args:
            source_docs (list): The list of annual reports in the analysis.
            gold_dir (str): The path of the gold summaries.
            candidate_dir (str): The path of the candidate summaries.

        """
        self.source_docs = source_docs
        self.gold_dir = gold_dir
        self.candidate_dir = candidate_dir

    def generate_noisy_summaries(
            self,
            doc_id: str,
            destructor: SummaryDestructor,
            noise_percentage: float
            ):
        """Generates noisy summaries for a given document.

        Args:
            doc_id (str): The number of the summary.
            destructor (SummaryDestructor): An object with summary destructive methods.

        """
        logger.info(f"Generating noisy summaries for {doc_id}...")

        try:
            noisy_summaries = {
                f'randomly_swapped_words_{noise_percentage}': destructor.random_swap_words(),
                f'consecutively_swapped_words_{noise_percentage}': destructor.consecutive_swap_words(),
                f'deleted_words_{noise_percentage}': destructor.remove_words(),
                f'removed_sentence_{noise_percentage}': destructor.remove_sentence(),
                f'inserted_sentence_{noise_percentage}': destructor.insert_sentence(target=doc_id, source_docs=self.source_docs, gold_dir=self.gold_dir),
                f'repeated_sentence_{noise_percentage}': destructor.repeat_sentence()
            }

            for summary_type, summary_content in noisy_summaries.items():
                file_path = os.path.join(self.candidate_dir, f"{doc_id}_{summary_type}.txt")
                with open(file_path, mode='w', encoding='utf-8') as file:
                    file.write(summary_content)
                logger.info(f"Saved {summary_type} summary for {doc_id} to {file_path}.")

        except Exception as e:
            logger.error(f"Failed to generate noisy summaries for {doc_id}: {e}")
            raise
