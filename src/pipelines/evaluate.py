import os

import torch
from dotenv import load_dotenv
from joblib import Parallel, delayed
from loguru import logger
from tqdm import tqdm

from src.modules.summary_evaluator import SummaryEvaluator

load_dotenv(override=True)
LANGUAGE = os.getenv("LANGUAGE")
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
TOTAL_CPU_CORES = os.cpu_count()
N_WORKERS = TOTAL_CPU_CORES // 2


def _process_doc_cpu(doc, gold_summaries_dir, candidate_summaries_dir, results_path):
    logger.info(f"Starting CPU-bound evaluation for {doc}")
    evaluator = SummaryEvaluator(
        source_docs=[doc], gold_dir=gold_summaries_dir, candidate_dir=candidate_summaries_dir, results_path=results_path
    )
    if doc not in evaluator._evaluated_docs_cpu:
        evaluator.evaluate_summaries(source_file=doc)
        logger.info(f"Finished CPU-bound evaluation for {doc}")
    else:
        logger.info(f"Skipping {doc} as it has already been evaluated.")


def run_cpu_metrics(source_docs, gold_summaries_dir, candidate_summaries_dir, results_path, n_workers=N_WORKERS):
    logger.info(f"Using {n_workers} CPU cores. Total CPU cores: {TOTAL_CPU_CORES}.")
    Parallel(n_jobs=n_workers)(
        delayed(_process_doc_cpu)(doc, gold_summaries_dir, candidate_summaries_dir, results_path)
        for doc in tqdm(source_docs, desc="Processing documents")
    )


def run_gpu_metrics(
    source_docs,
    source_dir,
    gold_summaries_dir,
    candidate_summaries_dir,
    results_path,
    no_refs,
    one_to_one,
    batch_size=BATCH_SIZE,
):
    logger.info(f"Starting GPU-bound evaluation with batch size {batch_size}.")
    evaluator = SummaryEvaluator(
        source_docs=source_docs,
        source_dir=source_dir,
        gold_dir=gold_summaries_dir,
        candidate_dir=candidate_summaries_dir,
        results_path=results_path,
    )
    if one_to_one:
        logger.info("Running one-to-one evaluation on GPU.")
        evaluator.evaluate_summaries_gpu_new()
    else:
        for doc in tqdm(source_docs, desc="Processing documents"):
            if doc not in evaluator._evaluated_docs_gpu:
                evaluator.evaluate_summaries_gpu_batch(source_file=doc, batch_size=batch_size, no_refs=no_refs)
                logger.info(f"Finished GPU-bound evaluation for {doc}")


def evaluate_summaries(
    source_docs,
    source_dir,
    gold_summaries_dir,
    candidate_summaries_dir,
    results_path,
    no_refs,
    one_to_one,
    run_cpu=True,
    run_gpu=True,
):
    logger.info("Starting evaluation pipeline")
    if run_cpu:
        run_cpu_metrics(source_docs, gold_summaries_dir, candidate_summaries_dir, results_path)
    if run_gpu:
        if not torch.cuda.is_available():
            logger.warning("No GPU available. Using CPU.")
        run_gpu_metrics(
            source_docs, source_dir, gold_summaries_dir, candidate_summaries_dir, results_path, no_refs, one_to_one
        )
    logger.info("Evaluation pipeline finished")
