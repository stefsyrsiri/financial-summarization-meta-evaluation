"""
Main script for running construction of noisy candidate summaries and scoring them.

This script initializes the DataCollector, SummaryCorruptor, SummaryGenerator,
and SummaryEvaluator classes to collect the Greek data, generates new noisy
summaries based on the Greek data and evaluates them against their original
version.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from argparse import ArgumentParser

from dotenv import load_dotenv
from loguru import logger

from pipelines.collect import collect_data
from pipelines.generate import generate_noisy_summaries
from pipelines.evaluate import evaluate_summaries


# Load from .env
load_dotenv(override=True)
LANGUAGE = os.getenv('LANGUAGE')
ANNUAL_REPORTS_DIR = os.getenv('ANNUAL_REPORTS_DIR')
GOLD_SUMMARIES_DIR = os.getenv('GOLD_SUMMARIES_DIR')
CANDIDATE_SUMMARIES_DIR = os.getenv('CANDIDATE_SUMMARIES_DIR')
RESULTS_PATH = os.getenv('RESULTS_PATH')
SUMMARY_VER = os.getenv('SUMMARY_VER')
FILE_EXTENSION = os.getenv('FILE_EXTENSION')

# TODO: Fix logger
# Configure logger
logger.add('logs/main_{time}.log', rotation='1 day', compression='zip', level='DEBUG', filter=lambda record: record["level"].name == "DEBUG")
logger.add('logs/errors_{time}.log', rotation='1 day', compression='zip', level='ERROR', filter=lambda record: record["level"].name == "ERROR")


@logger.catch
def main():
    logger.info(f'Starting the main process for {LANGUAGE}.')
    parser = ArgumentParser(description="Run pipeline steps.")

    # Parse args
    parser.add_argument('--collect', action='store_true', help="Collect data")
    parser.add_argument('--generate', action='store_true', help="Generate noisy summaries")
    parser.add_argument('--evaluate', action='store_true', help="Evaluate summaries")
    parser.add_argument('--all', action='store_true', help="Run all steps")

    # Subset of the source documents
    source_docs = [file[:-4] for file in os.listdir(ANNUAL_REPORTS_DIR)][:10]  # :-4 removes the file extension
    logger.info(f"Loaded {len(source_docs)} annual reports for noisy summary generation.")
    logger.info(f"Loaded {source_docs}.")

    args = parser.parse_args()

    if args.collect or args.all:
        collect_data()
    if args.generate or args.all:
        generate_noisy_summaries(
            source_docs=source_docs,
            gold_summaries_dir=GOLD_SUMMARIES_DIR,
            candidate_summaries_dir=CANDIDATE_SUMMARIES_DIR,
            summary_ver=SUMMARY_VER,
            file_extension=FILE_EXTENSION
            )
    if args.evaluate or args.all:
        evaluate_summaries(
            source_docs=source_docs,
            gold_summaries_dir=GOLD_SUMMARIES_DIR,
            candidate_summaries_dir=CANDIDATE_SUMMARIES_DIR,
            results_path=RESULTS_PATH
        )


if __name__ == '__main__':
    main()
