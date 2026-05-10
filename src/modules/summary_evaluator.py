import os
import re
import time
from filelock import FileLock

import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

from src.registries.languages_registry import LANGUAGES
from src.utils.summary_evaluator_utils import (
    load_metrics,
    load_checkpoint,
    get_candidate_filenames,
    get_candidate_metadata,
    load_candidate_texts,
    append_score,
)

load_dotenv()
SUMMARY_VER = os.getenv("SUMMARY_VER")
FILE_EXTENSION = os.getenv("FILE_EXTENSION")
LANGUAGE = os.getenv("LANGUAGE")


class SummaryEvaluator:
    def __init__(
        self,
        source_docs: list,
        source_dir: str = os.getenv("ANNUAL_REPORTS_DIR"),
        gold_dir: str = os.getenv("GOLD_SUMMARIES_DIR"),
        candidate_dir: str = os.getenv("CANDIDATE_SUMMARIES_DIR"),
        results_path: str = os.getenv("RESULTS_PATH"),
    ):
        self.language = LANGUAGES[LANGUAGE].code
        self.metrics = load_metrics(lang=self.language)
        self.source_docs = source_docs
        self.source_dir = source_dir
        self.gold_dir = gold_dir
        self.candidates_dir = candidate_dir
        self.results_path = results_path

        self.checkpoint_file_cpu = self.results_path + "_checkpoint_cpu.txt"
        self.checkpoint_file_gpu = self.results_path + "_checkpoint_gpu.txt"
        self._evaluated_docs_cpu = load_checkpoint(self.checkpoint_file_cpu)
        self._evaluated_docs_gpu = load_checkpoint(self.checkpoint_file_gpu)
        self.source_docs = [
            doc
            for doc in self.source_docs
            if doc not in self._evaluated_docs_cpu and doc not in self._evaluated_docs_gpu
        ]

        logger.info(
            f"SummaryEvaluator initialized. Evaluation metrics: {self.metrics.keys()}. "
            f"Source docs: {len(self.source_docs)}, "
            f"Already evaluated: {len(self._evaluated_docs_cpu)} (CPU), {len(self._evaluated_docs_gpu)} (GPU)."
        )

    def evaluate_summaries(self, source_file: str):
        logger.info(f"Evaluating source document: {source_file}")
        gold_summary_path = os.path.join(self.gold_dir, f"{source_file}{SUMMARY_VER}{FILE_EXTENSION}")
        data = {
            "source_doc": [],
            "eval_type": [],
            "eval_method": [],
            "variant": [],
            "score": [],
            "duration": [],
        }
        try:
            with open(gold_summary_path, mode="r", encoding="utf-8") as gold_f:
                gold_summary = gold_f.read()

            candidate_summaries = get_candidate_filenames(source_file, self.candidates_dir, self.gold_dir, randoms=True)

            for candidate_file in candidate_summaries:
                logger.info(f"Evaluating candidate summary: {candidate_file}")
                candidate_path, candidate_variant = get_candidate_metadata(self, candidate_file, source_file)

                try:
                    with open(candidate_path, mode="r", encoding="utf-8") as cand_f:
                        candidate_summary = cand_f.read()

                        # ----------N-GRAM
                        # Rouge 1
                        if "rouge1" in self.metrics:
                            start_time = time.time()
                            rouge1_result = self.metrics["rouge1"].score(
                                target=gold_summary, prediction=candidate_summary
                            )["rouge1"][2]  # 0: precision, 1: recall, 2: fmeasure
                            duration = time.time() - start_time
                            append_score(
                                data,
                                source_file=source_file,
                                type="N-gram-based",
                                method="ROUGE-1",
                                candidate_variant=candidate_variant,
                                result=rouge1_result,
                                duration=duration,
                            )

                        # Rouge 2
                        if "rouge2" in self.metrics:
                            start_time = time.time()
                            rouge2_result = self.metrics["rouge2"].score(
                                target=gold_summary, prediction=candidate_summary
                            )["rouge2"][2]
                            duration = time.time() - start_time
                            append_score(
                                data,
                                source_file=source_file,
                                type="N-gram-based",
                                method="ROUGE-2",
                                candidate_variant=candidate_variant,
                                result=rouge2_result,
                                duration=duration,
                            )

                        # ---------GRAPH
                        if {"autosummeng", "memog", "npower"} & self.metrics.keys():
                            start_time = time.time()
                            autosummeng_score, memog_score, npower_score = self.metrics["npower"].score(
                                target=gold_summary_path, prediction=candidate_path
                            )
                            duration = time.time() - start_time

                            # AutoSummENG
                            if "autosummeng" in self.metrics:
                                append_score(
                                    data,
                                    source_file=source_file,
                                    type="N-gram-graph-based",
                                    method="AutoSummENG",
                                    candidate_variant=candidate_variant,
                                    result=autosummeng_score,
                                    duration=duration,
                                )

                            # MeMoG
                            if "memog" in self.metrics:
                                append_score(
                                    data,
                                    source_file=source_file,
                                    type="N-gram-graph-based",
                                    method="MeMoG",
                                    candidate_variant=candidate_variant,
                                    result=memog_score,
                                    duration=duration,
                                )

                            # NPowER - computed in graph methods
                            if "npower" in self.metrics:
                                append_score(
                                    data,
                                    source_file=source_file,
                                    type="N-gram-graph-based",
                                    method="NPowER",
                                    candidate_variant=candidate_variant,
                                    result=npower_score,
                                    duration=duration,
                                )

                except FileNotFoundError as e:
                    logger.exception(f"File not found: {e}. Skipping candidate_file: {candidate_file}.")
                    continue

            # Append results to CSV with file locking (for multiprocessing safety)
            results_lock_path = self.results_path + "_cpu.lock"
            try:
                with FileLock(results_lock_path):
                    results_df = pd.DataFrame.from_dict(data, orient="index").transpose()
                    results_path_csv = self.results_path + ".csv"
                    results_df.to_csv(
                        results_path_csv,
                        mode="a",
                        header=not os.path.exists(results_path_csv),
                        index=False,
                    )

                # Save to checkpoint file to avoid re-evaluating
                checkpoint_lock_path = self.checkpoint_file_cpu + ".lock"
                with FileLock(checkpoint_lock_path):
                    with open(self.checkpoint_file_cpu, "a") as f:
                        f.write(f"{source_file}\n")

                logger.info(f"Evaluated {source_file} and saved results to {results_path_csv}.")
            except Exception as e:
                logger.exception(f"Failed to save results for {source_file}: {e}")
                return

        except FileNotFoundError as e:
            logger.exception(f"File not found: {e}. Skipping source_doc: {source_file}.")
            return

        if source_file == self.source_docs[-1] and os.path.exists(self.checkpoint_file_cpu):
            os.remove(self.checkpoint_file_cpu)
            os.remove(self.checkpoint_file_gpu)

        logger.info(f"Summary evaluation completed for document {source_file}.")

    def evaluate_summaries_gpu_batch(self, source_file: str, batch_size: int, no_refs: bool = False):
        logger.info(f"Evaluating source document: {source_file}")

        try:
            data = {
                "source_doc": [],
                "eval_type": [],
                "eval_method": [],
                "variant": [],
                "score": [],
                "duration": [],
            }
            candidate_summaries = get_candidate_filenames(source_file, self.candidates_dir, self.gold_dir, randoms=True)
            if no_refs:
                source_doc_path = os.path.join(self.source_dir, f"{source_file}{FILE_EXTENSION}")
                with open(source_doc_path, mode="r", encoding="utf-8") as source_f:
                    reference_doc = source_f.read()
            else:
                gold_summary_path = os.path.join(self.gold_dir, f"{source_file}{SUMMARY_VER}{FILE_EXTENSION}")
                with open(gold_summary_path, mode="r", encoding="utf-8") as gold_f:
                    reference_doc = gold_f.read()

            for i in range(0, len(candidate_summaries), batch_size):
                batch_files = candidate_summaries[i : i + batch_size]
                batch = load_candidate_texts(source_file, batch_files, self.gold_dir, self.candidates_dir)
                variants = [v for v, _ in batch]
                texts = [t for _, t in batch]

                # LongDocFACTScore batch
                if "ldfactscore" in self.metrics and self.language == "en":
                    start = time.time()
                    ldfact_score_scores = self.metrics["ldfactscore"].score_src_hyp_long(
                        srcs=[reference_doc] * len(texts), hyps=texts
                    )
                    duration = time.time() - start
                    for variant, score in zip(variants, ldfact_score_scores):
                        append_score(
                            data,
                            source_file=source_file,
                            type="NLI",
                            method="LongDocFACTScore",
                            candidate_variant=variant,
                            result=float(score),
                            duration=duration / len(batch),
                        )

                # BERTScore batch
                if "bertscore" in self.metrics:
                    start = time.time()
                    _, _, f1_scores = self.metrics["bertscore"].score(texts, [reference_doc] * len(texts))
                    duration = time.time() - start
                    for variant, score in zip(variants, f1_scores):
                        append_score(
                            data,
                            source_file=source_file,
                            type="Embeddings-based",
                            method="BERTScore",
                            candidate_variant=variant,
                            result=float(score),
                            duration=duration / len(batch),
                        )

                # BLEURT batch
                if "bleurt" in self.metrics:
                    start = time.time()
                    bleurt_scores = self.metrics["bleurt"].score(
                        references=[reference_doc] * len(texts), candidates=texts
                    )
                    duration = time.time() - start
                    for variant, score in zip(variants, bleurt_scores):
                        append_score(
                            data,
                            source_file=source_file,
                            type="NLI",
                            method="BLEURT",
                            candidate_variant=variant,
                            result=score,
                            duration=duration / len(batch),
                        )

                if self.language == "en":
                    # BARTScore batch
                    if "bartscore" in self.metrics:
                        start = time.time()
                        bart_scores = self.metrics["bartscore"].score(
                            texts,
                            [reference_doc] * len(texts),
                            batch_size=batch_size,
                        )
                        duration = time.time() - start
                        for variant, score in zip(variants, bart_scores):
                            append_score(
                                data,
                                source_file=source_file,
                                type="Embeddings-based",
                                method="BARTScore",
                                candidate_variant=variant,
                                result=score,
                                duration=duration / len(batch),
                            )

                    # FactCC batch
                    if "factcc" in self.metrics:
                        start = time.time()
                        factcc_logits, factcc_preds, factcc_probs = self.metrics["factcc"](
                            source_docs=[reference_doc] * len(texts),
                            summaries=texts,
                            batch_size=batch_size,
                        )
                        duration = time.time() - start
                        for variant, _, _, prob in zip(variants, factcc_logits, factcc_preds, factcc_probs):
                            append_score(
                                data,
                                source_file=source_file,
                                type="NLI",
                                method="FactCC",
                                candidate_variant=variant,
                                result=prob,
                                duration=duration / len(batch),
                            )

            results_lock_path = self.results_path + "_gpu.lock"
            checkpoint_lock_path = self.checkpoint_file_gpu + ".lock"

            try:
                with FileLock(results_lock_path):
                    results_df = pd.DataFrame.from_dict(data, orient="index").transpose()
                    results_path_csv = self.results_path + ".csv"
                    results_df.to_csv(
                        results_path_csv,
                        mode="a",
                        header=not os.path.exists(results_path_csv),
                        index=False,
                    )

                with FileLock(checkpoint_lock_path):
                    with open(self.checkpoint_file_gpu, "a") as f:
                        f.write(f"{source_file}\n")

                logger.info(f"Evaluated {source_file} (GPU) and saved results to {results_path_csv}.")
            except Exception as e:
                logger.exception(f"Failed to save GPU results for {source_file}: {e}")

        except FileNotFoundError as e:
            logger.exception(f"File not found: {e}. Skipping source_file: {source_file}.")
            return

        if source_file == self.source_docs[-1] and os.path.exists(self.checkpoint_file_gpu):
            os.remove(self.checkpoint_file_cpu)
            os.remove(self.checkpoint_file_gpu)
        logger.info(f"Summary evaluation completed for document {source_file}.")

    def evaluate_summaries_gpu_new(self):
        try:
            data = {
                "source_doc": [],
                "eval_type": [],
                "eval_method": [],
                "variant": [],
                "score": [],
                "duration": [],
            }
            for source_file in tqdm(self.source_docs, desc="Processing documents"):
                logger.info(f"Evaluating source document: {source_file}")
                source_doc_path = os.path.join(self.source_dir, f"{source_file}{FILE_EXTENSION}")
                with open(source_doc_path, mode="r", encoding="utf-8") as src_f:
                    reference_doc = src_f.read()

                if re.findall(r"^[\d]+$", source_file):
                    candidate_file = f"{source_file}{SUMMARY_VER}"
                    candidate_summary_path = os.path.join(self.gold_dir, f"{candidate_file}{FILE_EXTENSION}")
                else:
                    candidate_file = source_file
                    candidate_summary_path = os.path.join(self.candidates_dir, f"{candidate_file}{FILE_EXTENSION}")
                with open(candidate_summary_path, mode="r", encoding="utf-8") as cand_f:
                    candidate_summary = cand_f.read()

                if "bertscore" in self.metrics:
                    start = time.time()
                    _, _, f1_score = self.metrics["bertscore"].score([candidate_summary], [reference_doc])
                    duration = time.time() - start
                    append_score(
                        data,
                        source_file=source_file,
                        type="Embeddings-based",
                        method="BERTScore",
                        candidate_variant=candidate_file,
                        result=float(f1_score),
                        duration=duration,
                    )
                if "bartscore" in self.metrics:
                    start = time.time()
                    bart_score = self.metrics["bartscore"].score([candidate_summary], [reference_doc])[0]
                    duration = time.time() - start
                    append_score(
                        data,
                        source_file=source_file,
                        type="Embeddings-based",
                        method="BARTScore",
                        candidate_variant=candidate_file,
                        result=float(bart_score),
                        duration=duration,
                    )
                if "finsumeval" in self.metrics:
                    # Faithfulness
                    if "repeated_sentence" not in candidate_file:
                        start = time.time()
                        finsumeval_score = self.metrics["finsumeval"].score_faithfulness(
                            source=reference_doc, summary=candidate_summary
                        )
                        duration = time.time() - start
                        append_score(
                            data,
                            source_file=source_file,
                            type="NLI",
                            method="FinSumEval-faithfulness",
                            candidate_variant=candidate_file,
                            result=float(finsumeval_score.score),
                            duration=duration,
                        )

                    # Conciseness
                    if "inserted_sentence" not in candidate_file:
                        start = time.time()
                        finsumeval_score = self.metrics["finsumeval"].score_conciseness(summary=candidate_summary)
                        duration = time.time() - start
                        append_score(
                            data,
                            source_file=source_file,
                            type="Embeddings-based",
                            method="FinSumEval-conciseness",
                            candidate_variant=candidate_file,
                            result=float(finsumeval_score["score"]),
                            duration=duration,
                        )

                results_lock_path = self.results_path + "_gpu.lock"
                checkpoint_lock_path = self.checkpoint_file_gpu + ".lock"

                try:
                    with FileLock(results_lock_path):
                        results_df = pd.DataFrame.from_dict(data, orient="index").transpose()
                        results_path_csv = self.results_path + ".csv"
                        results_df.to_csv(
                            results_path_csv,
                            mode="a",
                            header=not os.path.exists(results_path_csv),
                            index=False,
                        )

                    with FileLock(checkpoint_lock_path):
                        with open(self.checkpoint_file_gpu, "a") as f:
                            f.write(f"{source_file}\n")

                    logger.info(f"Evaluated {source_file} (GPU) and saved results to {results_path_csv}.")
                except Exception as e:
                    logger.exception(f"Failed to save GPU results for {source_file}: {e}")

        except FileNotFoundError as e:
            logger.exception(f"File not found: {e}. Skipping source_file: {source_file}.")
            return

        if source_file == self.source_docs[-1] and os.path.exists(self.checkpoint_file_gpu):
            os.remove(self.checkpoint_file_cpu)
            os.remove(self.checkpoint_file_gpu)
        logger.info(f"Summary evaluation completed for document {source_file}.")
