import os

from dotenv import load_dotenv
from loguru import logger

from modules.summary_evaluator import SummaryEvaluator

load_dotenv(override=True)
LANGUAGE = os.getenv("LANGUAGE")


def evaluate_summaries(source_docs, gold_summaries_dir, candidate_summaries_dir, results_path):
    logger.info("Starting summaries evaluation.")
    evaluator = SummaryEvaluator(gold_dir=gold_summaries_dir, candidate_dir=candidate_summaries_dir)
    try:
        results = evaluator.evaluate_summaries(source_docs=source_docs)
        results.to_csv(results_path, index=False)
        logger.info(f"Successfully evaluated {LANGUAGE} data and saved results to {results_path}")
    except Exception as e:
        logger.error(f"Failed to evaluate the summaries: {e}")
