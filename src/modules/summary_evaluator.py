import os
import time
from filelock import FileLock

import pandas as pd
import torch
from bert_score import BERTScorer
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

from rouge_score.rouge_scorer import RougeScorer

# from evaluation_methods.MoverScore.moverscore import get_idf_dict, word_mover_score
from evaluation_methods.BARTScore.bart_score import BARTScorer
from evaluation_methods.Bleurt.bleurt.score import BleurtScorer
from evaluation_methods.NPowERV1 import npower
from src.utils.summary_evaluator_utils import append_score
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
    bleurt = BleurtScorer(checkpoint="evaluation_methods/Bleurt/bleurt/BLEURT-20")


class SummaryEvaluator:
    def __init__(
            self,
            source_docs: list,
            gold_dir: str,
            candidate_dir: str,
            results_path: str = os.getenv("RESULTS_PATH"),
            ):
        self.source_docs = source_docs
        self.gold_dir = gold_dir
        self.candidate_dir = candidate_dir
        self.results_path = results_path
        self.checkpoint_file = self.results_path + "_checkpoint.txt"
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, "r") as f:
                self._evaluated_docs = f.read().splitlines()
        else:
            self._evaluated_docs = []

        logger.info(f"SummaryEvaluator initialized. Source docs: {len(self.source_docs)}, Already evaluated: {len(self._evaluated_docs)}.")

    def evaluate_summaries(self):
        logger.info("Starting summary evaluation.")

        self.source_docs = [doc for doc in self.source_docs if doc not in self._evaluated_docs]
        for source_file in tqdm(self.source_docs, desc="Processing documents"):
            logger.info(f"Evaluating source document: {source_file}")
            source_path = os.path.join(self.gold_dir, f"{source_file}{SUMMARY_VER}{FILE_EXTENSION}")
            data = {
                "source_doc": [],
                "eval_type": [],
                "eval_method": [],
                "variant": [],
                "score": [],
                "duration": []
            }
            try:
                with open(source_path, mode="r", encoding="utf-8") as gold_f:
                    gold_summary = gold_f.read()

                # Actual candidate summaries
                candidate_summaries = [doc for doc in os.listdir(self.candidate_dir) if doc.startswith(f"{source_file}_")]

                # 10 randoms
                other_summaries = [doc for doc in os.listdir(self.gold_dir) if not doc.startswith(f"{source_file}_") and doc.endswith(f"{SUMMARY_VER}{FILE_EXTENSION}")]
                candidate_summaries.extend(other_summaries[:10])

                # Source summary
                candidate_summaries.insert(0, source_file)

                for candidate_file in candidate_summaries:
                    logger.info(f"Evaluating candidate summary: {candidate_file}")
                    # Source
                    if candidate_file == source_file:
                        candidate_path = os.path.join(self.gold_dir, f"{candidate_file}{SUMMARY_VER}{FILE_EXTENSION}")
                        candidate_variant = "source"
                    # Other (gold) summaries
                    elif candidate_file.endswith(f"{SUMMARY_VER}{FILE_EXTENSION}"):
                        candidate_path = os.path.join(self.gold_dir, candidate_file)
                        candidate_variant = candidate_file.removesuffix(f"{FILE_EXTENSION}")
                    # Candidate / Destroyed summaries
                    else:
                        candidate_path = os.path.join(self.candidate_dir, candidate_file)
                        candidate_variant = candidate_file.removeprefix(f"{source_file}_").removesuffix(f"{FILE_EXTENSION}")

                    try:
                        with open(candidate_path, mode="r", encoding="utf-8") as cand_f:
                            candidate_summary = cand_f.read()

                            # ----------N-GRAM
                            # Rouge 1
                            start_time = time.time()
                            rouge1_result = rouge1.score(target=gold_summary, prediction=candidate_summary)["rouge1"][2]  # 0: precision, 1: recall, 2: fmeasure
                            duration = time.time() - start_time
                            append_score(data, source_file=source_file, type="N-gram", method="Rouge1", candidate_variant=candidate_variant, result=rouge1_result, duration=duration)

                            # Rouge 2
                            start_time = time.time()
                            rouge2_result = rouge2.score(target=gold_summary, prediction=candidate_summary)["rouge2"][2]
                            duration = time.time() - start_time
                            append_score(data, source_file=source_file, type="N-gram", method="Rouge2", candidate_variant=candidate_variant, result=rouge2_result, duration=duration)

                            # ---------GRAPH
                            start_time = time.time()
                            autosummeng_score, memog_score, npower_score = npower.score(target=source_path, prediction=candidate_path)
                            duration = time.time() - start_time

                            # AutoSummENG
                            append_score(data, source_file=source_file, type="Graph-based", method="AutoSummENG", candidate_variant=candidate_variant, result=autosummeng_score, duration=duration)

                            # MeMoG
                            append_score(data, source_file=source_file, type="Graph-based", method="MeMoG", candidate_variant=candidate_variant, result=memog_score, duration=duration)

                            # ---------META
                            # NPowER - computed in graph methods
                            append_score(data, source_file=source_file, type="Meta", method="NPowER", candidate_variant=candidate_variant, result=npower_score, duration=duration)

                            # ---------EMBEDDINGS-BASED
                            # BERTScore
                            start_time = time.time()
                            P, R, F1 = bertscore.score([candidate_summary], [gold_summary])
                            duration = time.time() - start_time
                            append_score(data, source_file=source_file, type="Embeddings-based", method="BERTScore", candidate_variant=candidate_variant, result=float(F1), duration=duration)

                            if LANGUAGE_CODE == "en":

                                # BARTscore
                                start_time = time.time()
                                bartscore_result = bartscore.score([candidate_summary], [gold_summary], batch_size=4)
                                duration = time.time() - start_time
                                append_score(data, source_file=source_file, type="Embeddings-based", method="BARTScore", candidate_variant=candidate_variant, result=bartscore_result[0], duration=duration)

                                # Bleurt
                                start_time = time.time()
                                bleurt_result = bleurt.score(references=[gold_summary], candidates=[candidate_summary])
                                duration = time.time() - start_time
                                append_score(data, source_file=source_file, type="Embeddings-based", method="Bleurt", candidate_variant=candidate_variant, result=bleurt_result[0], duration=duration)

                    except FileNotFoundError as e:
                        logger.exception(f"File not found: {e}. Skipping candidate_file: {candidate_file}.")
                        continue

                # Append results to CSV with file locking (for multiprocessing safety)
                results_lock_path = self.results_path + ".lock"
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
                    checkpoint_lock_path = self.checkpoint_file + ".lock"
                    with FileLock(checkpoint_lock_path):
                        with open(self.checkpoint_file, "a") as f:
                            f.write(f"{source_file}\n")

                    logger.info(f"Evaluated {source_file} and saved results to {results_path_csv}.")
                except Exception as e:
                    logger.exception(f"Failed to save results for {source_file}: {e}")
                    continue

            except FileNotFoundError as e:
                logger.exception(f"File not found: {e}. Skipping source_doc: {source_file}.")
                continue

        with open(self.checkpoint_file, "r") as f:
            evaluated_docs = f.read().splitlines()
        logger.info(f"Evaluated documents: {len(evaluated_docs)} ({len(evaluated_docs)/len(self.source_docs):.2%}).")

        # Clean up checkpoint file
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)

        logger.info("Summary evaluation completed.")
