"""
Main script for running construction of noisy candidate summaries and scoring them.

This script initializes the DataCollector, SummaryDestructor, SummaryGenerator,
and SummaryEvaluator classes to collect the Greek data, generates new noisy
summaries based on the Greek data and evaluates them against their original
version.
"""

import os

from dotenv import load_dotenv
from loguru import logger
import numpy as np

from data_collector import DataCollector
from summary_destructor import SummaryDestructor
from summary_generator import SummaryGenerator
from summary_evaluator import SummaryEvaluator


# Load from .env
load_dotenv()
ANNUAL_REPORTS_DIR = os.getenv('ANNUAL_REPORTS_DIR')
GOLD_SUMMARIES_DIR = os.getenv('GOLD_SUMMARIES_DIR')
CANDIDATE_SUMMARIES_DIR = os.getenv('CANDIDATE_SUMMARIES_DIR')
RESULTS_PATH = os.getenv('RESULTS_PATH')

# Configure logger
logger.add('logs/main_{time}.log', rotation='1 day', compression='zip', level='DEBUG')


def main():
    logger.info('Starting the main process.')

    # Collect data for destruction and evaluation
    data_collector = DataCollector()
    logger.info('Collecting data.')
    try:
        data_collector.collect_data()
        logger.info('Successfully collected Greek data.')
    except Exception as e:
        logger.error(f'Failed data collection: {e}')
        return

    # Subset of the source documents
    source_docs = [file[:-4] for file in os.listdir(ANNUAL_REPORTS_DIR)][:10]
    logger.info(f"Loaded {len(source_docs)} annual reports for noisy summary generation.")
    logger.info(f"Loaded {source_docs}.")

    # Noisy summaries
    summary_generator = SummaryGenerator(source_docs=source_docs, gold_dir=GOLD_SUMMARIES_DIR, candidate_dir=CANDIDATE_SUMMARIES_DIR)

    for doc in source_docs:
        logger.info(f"Processing annual report '{doc}' for noisy summary generation.")
        try:
            with open(file=os.path.join(GOLD_SUMMARIES_DIR, f'{doc}_1.txt'), mode='r', encoding='utf-8') as file:
                gold_summary = file.read()

            # Summary Destruction
            destructor = SummaryDestructor(input_summary=gold_summary, noise_percentage=0.9)

            # Summary Generation
            percentages = np.round(np.linspace(0.9, 0.1, num=5), 1).tolist()
            for percentage in percentages:
                destructor.noise_percentage = percentage
                summary_generator.generate_noisy_summaries(doc_id=doc, destructor=destructor, noise_percentage=percentage)
        except Exception as e:
            logger.error(f"Failed processing for gold summary {doc}: {e}")

    # Summaries evaluation
    logger.info("Starting summaries evaluation.")
    evaluator = SummaryEvaluator(gold_dir=GOLD_SUMMARIES_DIR, candidate_dir=CANDIDATE_SUMMARIES_DIR)
    try:
        results = evaluator.evaluate_summaries(source_docs=source_docs)
        results.to_csv(RESULTS_PATH, index=False)
        logger.info(f"Successfully evaluated Greek data and saved results to {RESULTS_PATH}")
    except Exception as e:
        logger.error(f"Failed to evaluate the summaries: {e}")


if __name__ == '__main__':
    main()
