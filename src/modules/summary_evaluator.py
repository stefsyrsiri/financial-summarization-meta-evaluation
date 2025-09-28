import os
import time
from filelock import FileLock

import pandas as pd
import torch
from bert_score import BERTScorer
from dotenv import load_dotenv
from loguru import logger
from longdocfactscore.ldfacts import LongDocFACTScore
from rouge_score.rouge_scorer import RougeScorer
from summac.model_summac import SummaCZS

from evaluation_methods.BARTScore.bart_score import BARTScorer
from evaluation_methods.Bleurt.bleurt.score import BleurtScorer
from evaluation_methods.NPowERV1 import npower
from evaluation_methods.FactCC.factcc import batched_FactCC
from src.utils.summary_evaluator_utils import (
    get_candidate_filenames,
    get_candidate_metadata,
    load_candidate_texts,
    append_score
)
from src.modules.tokenizer import Tokenizer

load_dotenv()
SUMMARY_VER = os.getenv("SUMMARY_VER")
FILE_EXTENSION = os.getenv("FILE_EXTENSION")
LANGUAGE_CODE = os.getenv("LANGUAGE_CODE")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
tokenizer = Tokenizer(lang_code=LANGUAGE_CODE)
rouge1 = RougeScorer(["rouge1"], use_stemmer=False, tokenizer=tokenizer)
rouge2 = RougeScorer(["rouge2"], use_stemmer=False, tokenizer=tokenizer)
bertscore = BERTScorer(lang=LANGUAGE_CODE, device=device)  # bert-base-multilingual-cased or deberta-xlarge-mnli

if LANGUAGE_CODE == "en":
    bartscore = BARTScorer(device=device, checkpoint="facebook/bart-large-cnn")
    ldfact_score = LongDocFACTScore(device=device)
    bleurt = BleurtScorer(checkpoint="evaluation_methods/Bleurt/bleurt/BLEURT-20")
    summaczs = SummaCZS(granularity="sentence", model_name="vitc", device=device)


class SummaryEvaluator:
    def __init__(
            self,
            source_docs: list,
            source_dir: str = os.getenv("ANNUAL_REPORTS_DIR"),
            gold_dir: str = os.getenv("GOLD_SUMMARIES_DIR"),
            candidate_dir: str = os.getenv("CANDIDATE_SUMMARIES_DIR"),
            results_path: str = os.getenv("RESULTS_PATH"),
            ):
        self.source_docs = source_docs
        self.source_dir = source_dir
        self.gold_dir = gold_dir
        self.candidates_dir = candidate_dir
        self.results_path = results_path
        self.checkpoint_file_cpu = self.results_path + "_checkpoint_cpu.txt"
        if os.path.exists(self.checkpoint_file_cpu):
            with open(self.checkpoint_file_cpu, "r") as f:
                self._evaluated_docs_cpu = f.read().splitlines()
                self.source_docs = [doc for doc in self.source_docs if doc not in self._evaluated_docs_cpu]
        else:
            self._evaluated_docs_cpu = []
        self.checkpoint_file_gpu = self.results_path + "_checkpoint_gpu.txt"
        if os.path.exists(self.checkpoint_file_gpu):
            with open(self.checkpoint_file_gpu, "r") as f:
                self._evaluated_docs_gpu = f.read().splitlines()
                self.source_docs = [doc for doc in self.source_docs if doc not in self._evaluated_docs_gpu]
        else:
            self._evaluated_docs_gpu = []
        self._rouge1 = rouge1
        self._rouge2 = rouge2
        self._npower = npower
        self._bertscore = bertscore
        self._bartscore = bartscore if LANGUAGE_CODE == "en" else None
        self._ldfact_score = ldfact_score if LANGUAGE_CODE == "en" else None
        self._bleurt = bleurt if LANGUAGE_CODE == "en" else None
        self._factcc = batched_FactCC if LANGUAGE_CODE == "en" else None
        self._summaczs = summaczs if LANGUAGE_CODE == "en" else None

        logger.info(f"SummaryEvaluator initialized. Source docs: {len(self.source_docs)}, Already evaluated: {len(self._evaluated_docs_cpu)} (CPU), {len(self._evaluated_docs_gpu)} (GPU).")

    def evaluate_summaries(self, source_file: str):
        logger.info(f"Evaluating source document: {source_file}")
        gold_summary_path = os.path.join(self.gold_dir, f"{source_file}{SUMMARY_VER}{FILE_EXTENSION}")
        data = {
            "source_doc": [],
            "eval_type": [],
            "eval_method": [],
            "variant": [],
            "score": [],
            "duration": []
            }
        try:
            with open(gold_summary_path, mode="r", encoding="utf-8") as gold_f:
                gold_summary = gold_f.read()

            candidate_summaries = get_candidate_filenames(self, source_file)

            for candidate_file in candidate_summaries:
                logger.info(f"Evaluating candidate summary: {candidate_file}")
                candidate_path, candidate_variant = get_candidate_metadata(self, candidate_file, source_file)

                try:
                    with open(candidate_path, mode="r", encoding="utf-8") as cand_f:
                        candidate_summary = cand_f.read()

                        # ----------N-GRAM
                        # Rouge 1
                        start_time = time.time()
                        rouge1_result = self._rouge1.score(target=gold_summary, prediction=candidate_summary)["rouge1"][2]  # 0: precision, 1: recall, 2: fmeasure
                        duration = time.time() - start_time
                        append_score(data, source_file=source_file, type="N-gram", method="Rouge1", candidate_variant=candidate_variant, result=rouge1_result, duration=duration)

                        # Rouge 2
                        start_time = time.time()
                        rouge2_result = self._rouge2.score(target=gold_summary, prediction=candidate_summary)["rouge2"][2]
                        duration = time.time() - start_time
                        append_score(data, source_file=source_file, type="N-gram", method="Rouge2", candidate_variant=candidate_variant, result=rouge2_result, duration=duration)

                        # ---------GRAPH
                        start_time = time.time()
                        autosummeng_score, memog_score, npower_score = self._npower.score(target=gold_summary_path, prediction=candidate_path)
                        duration = time.time() - start_time

                        # AutoSummENG
                        append_score(data, source_file=source_file, type="Graph-based", method="AutoSummENG", candidate_variant=candidate_variant, result=autosummeng_score, duration=duration)

                        # MeMoG
                        append_score(data, source_file=source_file, type="Graph-based", method="MeMoG", candidate_variant=candidate_variant, result=memog_score, duration=duration)

                        # ---------META
                        # NPowER - computed in graph methods
                        append_score(data, source_file=source_file, type="Meta", method="NPowER", candidate_variant=candidate_variant, result=npower_score, duration=duration)

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
                        index=False
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
                "duration": []
            }
            candidate_summaries = get_candidate_filenames(self, source_file)

            if no_refs:
                source_doc_path = os.path.join(self.source_dir, f"{source_file}{FILE_EXTENSION}")
                with open(source_doc_path, mode="r", encoding="utf-8") as source_f:
                    source_doc = source_f.read()
            else:
                gold_summary_path = os.path.join(self.gold_dir, f"{source_file}{SUMMARY_VER}{FILE_EXTENSION}")
                with open(gold_summary_path, mode="r", encoding="utf-8") as gold_f:
                    gold_summary = gold_f.read()

            for i in range(0, len(candidate_summaries), batch_size):
                batch_files = candidate_summaries[i:i+batch_size]
                batch = load_candidate_texts(self, source_file, batch_files)
                variants = [v for v, _ in batch]
                texts = [t for _, t in batch]

                if no_refs:
                    # LongDocFACTScore batch
                    start = time.time()
                    ldfact_score_scores = self._ldfact_score.score_src_hyp_long(srcs=[source_doc]*len(texts), hyps=texts)
                    duration = time.time() - start
                    for variant, score in zip(variants, ldfact_score_scores):
                        append_score(data, source_file=source_file, type="Embeddings-based", method="LongDocFACTScore", candidate_variant=variant, result=float(score), duration=duration/len(batch))
                else:
                    # BERTScore batch
                    start = time.time()
                    _, _, f1_scores = self._bertscore.score(texts, [gold_summary]*len(texts))
                    duration = time.time() - start
                    for variant, score in zip(variants, f1_scores):
                        append_score(data, source_file=source_file, type="Embeddings-based", method="BERTScore", candidate_variant=variant, result=float(score), duration=duration/len(batch))

                    if LANGUAGE_CODE == "en":
                        # BARTScore batch
                        start = time.time()
                        bart_scores = self._bartscore.score(texts, [gold_summary]*len(texts), batch_size=batch_size)
                        duration = time.time() - start
                        for variant, score in zip(variants, bart_scores):
                            append_score(data, source_file=source_file, type="Embeddings-based", method="BARTScore", candidate_variant=variant, result=score, duration=duration/len(batch))

                        # BLEURT batch
                        start = time.time()
                        bleurt_scores = self._bleurt.score(references=[gold_summary]*len(texts), candidates=texts)
                        duration = time.time() - start
                        for variant, score in zip(variants, bleurt_scores):
                            append_score(data, source_file=source_file, type="Embeddings-based", method="Bleurt", candidate_variant=variant, result=score, duration=duration/len(batch))

                        # FactCC batch
                        start = time.time()
                        factcc_logits, factcc_preds, factcc_probs = self._factcc(source_docs=[gold_summary]*len(texts), summaries=texts, batch_size=batch_size)
                        duration = time.time() - start
                        for variant, _, _, prob in zip(variants, factcc_logits, factcc_preds, factcc_probs):
                            append_score(data, source_file=source_file, type="Embeddings-based", method="FactCC", candidate_variant=variant, result=prob, duration=duration/len(batch))

                        # SummacZS batch
                        start = time.time()
                        summaczs_scores = self._summaczs.score(sources=[gold_summary]*len(texts), generateds=texts, batch_size=batch_size)["scores"]
                        duration = time.time() - start
                        for variant, score in zip(variants, summaczs_scores):
                            append_score(data, source_file=source_file, type="Embeddings-based", method="SummaCZS", candidate_variant=variant, result=score, duration=duration/len(batch))

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
                        index=False
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
            os.remove(self.checkpoint_file_gpu)
        logger.info(f"Summary evaluation completed for document {source_file}.")
