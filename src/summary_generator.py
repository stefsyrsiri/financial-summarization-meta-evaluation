import os
from loguru import logger
from summary_destructor import SummaryDestructor


class SummaryGenerator:
    def __init__(self, source_docs, gold_dir, candidate_dir):
        self.source_docs = source_docs
        self.gold_dir = gold_dir
        self.candidate_dir = candidate_dir

    def generate_noisy_summaries(self, doc_name: str, doc_content: str, destructor: SummaryDestructor):
        """Generates noisy summaries for a given document."""
        logger.info(f"Generating noisy summaries for {doc_name}...")

        try:
            noisy_summaries = {
                'shuffled_words': destructor.shuffle_words(),
                'deleted_words': destructor.remove_words(n_words=10),
                'removed_sentence': destructor.remove_sentence(),
                'inserted_sentence': destructor.insert_sentence(target=doc_name, source_docs=self.source_docs, gold_dir=self.gold_dir),
                'repeated_sentence': destructor.repeat_sentence(n_repeats=10)
            }

            for summary_type, summary_content in noisy_summaries.items():
                file_path = os.path.join(self.candidate_dir, f"{doc_name}_{summary_type}.txt")
                with open(file_path, mode='w', encoding='utf-8') as file:
                    file.write(summary_content)
                logger.info(f"Saved {summary_type} summary for {doc_name} to {file_path}.")

        except Exception as e:
            logger.error(f"Failed to generate noisy summaries for {doc_name}: {e}")
            raise
