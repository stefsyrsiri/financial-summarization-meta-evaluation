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
from transformers import BertTokenizer
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()  # Huggingface warnings

from evaluation_methods.FinSumEval.metric.extractors.ngram import NgramExtractor
from evaluation_methods.FinSumEval.metric.tokenizers.tokenizer import SpacyTokenizer

from src.modules.data_collector import DataCollector
from src.modules.stats_extractor import StatsExtractor
from src.modules.tokenizer import Tokenizer

from src.pipelines.generate import generate_gold_summaries, generate_noisy_summaries
from src.pipelines.evaluate import evaluate_summaries
from src.utils.sampling import get_sample_docs

from src.registries.languages_registry import LANGUAGES


load_dotenv(override=True)
LANGUAGE = os.getenv("LANGUAGE")
LANGUAGE_CODE = LANGUAGES[LANGUAGE].code
ANNUAL_REPORTS_DIR = os.getenv("ANNUAL_REPORTS_DIR")
GOLD_SUMMARIES_DIR = os.getenv("GOLD_SUMMARIES_DIR")
EXTRACTED_SUMMARIES_DIR = os.getenv("EXTRACTED_SUMMARIES_DIR")
CANDIDATE_SUMMARIES_DIR = os.getenv("CANDIDATE_SUMMARIES_DIR")
RESULTS_PATH = os.getenv("RESULTS_PATH")
SUMMARY_VER = os.getenv("SUMMARY_VER")
FILE_EXTENSION = os.getenv("FILE_EXTENSION")

N_SAMPLES = int(os.getenv("N_SAMPLES", 5))
SAMPLE_K_DOCS = int(os.getenv("SAMPLE_K_DOCS"))
SAMPLED_DOCS_PATH = os.getenv("SAMPLED_DOCS_PATH")
SEEDS_PATH = os.getenv("SEEDS_PATH")

DATASET_PATH = os.getenv("DATASET_PATH")
STATISTICS_PATH = os.getenv("STATISTICS_PATH")

# Silence the logger
logger.remove()

# Terminal
logger.add(sys.stdout, level="INFO", enqueue=True)

# File
logger.add(
    "logs/main_{time}.log",
    rotation="1 day",
    compression="zip",
    level="INFO",
    retention="7 days",
)


@logger.catch
def main():
    logger.info(f"Starting the main process for {LANGUAGE}.")

    parser = ArgumentParser(description="Run pipeline steps.")

    # Parse args
    # Data collection
    parser.add_argument("--collect", action="store_true", help="Collect data.")

    # Get document statistics
    parser.add_argument(
        "--merge-datasets",
        action="store_true",
        help="Create a unified dataset from all the .txt files.",
    )
    parser.add_argument("--stats", action="store_true", help="Get text statistics.")

    # Sampling
    parser.add_argument("--sample", action="store_true", help="Sample source documents.")

    # Noisy summaries generation
    parser.add_argument("--generate", action="store_true", help="Generate noisy summaries.")
    parser.add_argument("--truncate", action="store_true", help="Truncate long documents.")

    # Evaluation
    parser.add_argument("--evaluate", action="store_true", help="Evaluate summaries.")
    parser.add_argument("--cpu", action="store_true", help="Run only CPU-bound evaluation.")
    parser.add_argument("--gpu", action="store_true", help="Run only GPU-bound evaluation.")
    parser.add_argument("--no-refs", action="store_true", help="Reference free evaluation.")
    parser.add_argument("--new", action="store_true", help="Evaluate using extracted summaries.")

    # N-gram overlap-based extraction
    parser.add_argument("--ngram-extract", action="store_true", help="Run n-gram overlap-based extraction.")

    # Subset
    parser.add_argument("--subset", type=int, help="Subset of source documents to process.")

    # Run all steps
    parser.add_argument("--all", action="store_true", help="Run all steps.")
    args = parser.parse_args()

    # Subset of the source documents
    if args.new:
        source_docs = [file[:-4] for file in os.listdir(EXTRACTED_SUMMARIES_DIR)]  # :-4 removes the file extension
    elif LANGUAGE_CODE == "en":
        source_docs = [file[:-4] for file in os.listdir(ANNUAL_REPORTS_DIR)]  # :-4 removes the file extension
        logger.debug(f"Main path: {SAMPLED_DOCS_PATH}")
        source_docs = get_sample_docs(
            sampled_docs_path=SAMPLED_DOCS_PATH,
            seeds_path=SEEDS_PATH,
            source_docs=source_docs,
            n_samples=N_SAMPLES,
            sample_k=SAMPLE_K_DOCS,
        )
    else:
        source_docs = [file[:-4] for file in os.listdir(ANNUAL_REPORTS_DIR)]  # :-4 removes the file extension

    source_docs = source_docs[: args.subset] if args.subset else source_docs

    # Optional sampling
    if args.sample:
        logger.debug(f"Main path: {SAMPLED_DOCS_PATH}")
        source_docs = get_sample_docs(
            sampled_docs_path=SAMPLED_DOCS_PATH,
            seeds_path=SEEDS_PATH,
            source_docs=source_docs,
            n_samples=N_SAMPLES,
            sample_k=SAMPLE_K_DOCS,
        )

    logger.info(f"Running process on {len(source_docs)} annual reports.")
    logger.debug(f"Source documents: {source_docs}")

    if args.collect or args.all:
        data_collector = DataCollector()
        data_collector.collect()

    if args.merge_datasets:
        StatsExtractor.get_dataset()

    if args.stats:
        df = StatsExtractor.get_dataset(dataset_path=DATASET_PATH)
        spacy_tokenizer = Tokenizer(lang_code=LANGUAGE_CODE)
        bert_tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
        stats_extractor = StatsExtractor(
            dataset_path=DATASET_PATH,
            spacy_tokenizer=spacy_tokenizer,
            bert_tokenizer=bert_tokenizer,
            results_path=STATISTICS_PATH,
        )
        stats_extractor.get_stats(df=df)

    if args.generate or args.all:
        generate_noisy_summaries(
            source_docs=source_docs,
            gold_summaries_dir=GOLD_SUMMARIES_DIR,
            candidate_summaries_dir=CANDIDATE_SUMMARIES_DIR,
            summary_ver=SUMMARY_VER,
            file_extension=FILE_EXTENSION,
            truncate_for_bert=args.truncate,
        )

    if args.ngram_extract:
        tokenizer = SpacyTokenizer(LANGUAGE_CODE)
        extractor = NgramExtractor(tokenizer)
        generate_gold_summaries(
            source_docs=source_docs,
            source_dir=ANNUAL_REPORTS_DIR,
            candidate_summaries_dir=CANDIDATE_SUMMARIES_DIR,
            gold_summaries_dir=GOLD_SUMMARIES_DIR,
            extracted_summaries_dir=EXTRACTED_SUMMARIES_DIR,
            file_extension=FILE_EXTENSION,
            extractor=extractor,
        )

    if args.evaluate or args.all:
        evaluate_summaries(
            source_docs=source_docs,
            source_dir=EXTRACTED_SUMMARIES_DIR if args.new else ANNUAL_REPORTS_DIR,
            gold_summaries_dir=GOLD_SUMMARIES_DIR,
            candidate_summaries_dir=CANDIDATE_SUMMARIES_DIR,
            results_path=RESULTS_PATH,
            no_refs=args.no_refs,
            run_cpu=args.cpu or (not args.cpu and not args.gpu),
            run_gpu=args.gpu or (not args.cpu and not args.gpu),
            one_to_one=args.new,
        )


if __name__ == "__main__":
    main()
