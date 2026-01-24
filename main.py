"""
Main script for running construction of noisy candidate summaries and scoring them.

Initializes the DataCollector, SummaryCorruptor, SummaryGenerator,
and SummaryEvaluator classes to collect the Greek data, generates new noisy
summaries based on the Greek data and evaluates them against their original
version.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # TensorFlow suppress info/warning
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="spacy")

from argparse import ArgumentParser
from dotenv import load_dotenv
from loguru import logger
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()  # Huggingface warnings

from src.pipelines.collect import collect_data
from src.pipelines.generate import generate_noisy_summaries
from src.pipelines.evaluate import evaluate_summaries
from src.utils.sampling import get_sample_docs


load_dotenv(override=True)
LANGUAGE = os.getenv("LANGUAGE")
ANNUAL_REPORTS_DIR = os.getenv("ANNUAL_REPORTS_DIR")
GOLD_SUMMARIES_DIR = os.getenv("GOLD_SUMMARIES_DIR")
CANDIDATE_SUMMARIES_DIR = os.getenv("CANDIDATE_SUMMARIES_DIR")
RESULTS_PATH = os.getenv("RESULTS_PATH")
SUMMARY_VER = os.getenv("SUMMARY_VER")
FILE_EXTENSION = os.getenv("FILE_EXTENSION")
N_SAMPLES = int(os.getenv("N_SAMPLES", 5))
SAMPLED_DOCS_PATH = os.getenv("SAMPLED_DOCS_PATH")
SEEDS_PATH = os.getenv("SEEDS_PATH")

# Silence the logger
logger.remove()

# Terminal
logger.add(sys.stdout, level="INFO", enqueue=True)

# File
logger.add("logs/main_{time}.log", rotation="1 day", compression="zip", level="INFO", retention="7 days")


@logger.catch
def main():
    logger.info(f"Starting the main process for {LANGUAGE}.")

    parser = ArgumentParser(description="Run pipeline steps.")

    # Parse args
    # Data collection
    parser.add_argument("--collect", action="store_true", help="Collect data")

    # Noisy summaries generation
    parser.add_argument("--generate", action="store_true", help="Generate noisy summaries")
    parser.add_argument("--truncate", action="store_true", help="Truncate long documents")

    # Evaluation
    parser.add_argument("--evaluate", action="store_true", help="Evaluate summaries")
    parser.add_argument("--cpu", action="store_true", help="Run only CPU-bound evaluation")
    parser.add_argument("--gpu", action="store_true", help="Run only GPU-bound evaluation")
    parser.add_argument("--no-refs", action="store_true", help="Reference free evaluation")

    # Run all steps
    parser.add_argument("--all", action="store_true", help="Run all steps")

    # Subset of the source documents
    source_docs = [file[:-4] for file in os.listdir(ANNUAL_REPORTS_DIR)]  # :-4 removes the file extension

    # Sampling for English docs
    if LANGUAGE == "English":
        logger.debug(f"Main path: {SAMPLED_DOCS_PATH}")
        source_docs = get_sample_docs(
            sampled_docs_path=SAMPLED_DOCS_PATH,
            seeds_path=SEEDS_PATH,
            source_docs=source_docs,
            n_samples=N_SAMPLES,
            )

    logger.info(f"Running process on {len(source_docs)} annual reports.")
    logger.debug(f"Source documents: {source_docs}")

    args = parser.parse_args()

    if args.collect or args.all:
        collect_data()
    if args.generate or args.all:
        generate_noisy_summaries(
            source_docs=source_docs,
            gold_summaries_dir=GOLD_SUMMARIES_DIR,
            candidate_summaries_dir=CANDIDATE_SUMMARIES_DIR,
            summary_ver=SUMMARY_VER,
            file_extension=FILE_EXTENSION,
            truncate_for_bert=args.truncate
            )
    if args.evaluate or args.all:
        evaluate_summaries(
            source_docs=source_docs,
            source_dir=ANNUAL_REPORTS_DIR,
            gold_summaries_dir=GOLD_SUMMARIES_DIR,
            candidate_summaries_dir=CANDIDATE_SUMMARIES_DIR,
            results_path=RESULTS_PATH,
            no_refs=args.no_refs,
            run_cpu=args.cpu or (not args.cpu and not args.gpu),
            run_gpu=args.gpu or (not args.cpu and not args.gpu)
        )


if __name__ == "__main__":
    main()
