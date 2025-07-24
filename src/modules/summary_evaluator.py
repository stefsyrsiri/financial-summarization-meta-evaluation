import os
import time

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
RESULTS_PATH = os.getenv("RESULTS_PATH")

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
            gold_dir: str,
            candidate_dir: str,
            ):
        self.gold_dir = gold_dir
        self.candidate_dir = candidate_dir
        self.data = {
            "source_doc": [],
            "eval_type": [],
            "eval_method": [],
            "variant": [],
            "score": [],
            "duration": [],
        }
        logger.info("SummaryEvaluator initialized.")

    def evaluate_summaries(self, source_docs):
        logger.info("Starting summary evaluation.")
        for source_file in tqdm(source_docs, desc="Processing documents"):
            source_path = os.path.join(self.gold_dir, f"{source_file}{SUMMARY_VER}{FILE_EXTENSION}")
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

                    with open(candidate_path, mode="r", encoding="utf-8") as cand_f:
                        candidate_summary = cand_f.read()

                        # ----------N-GRAM
                        # Rouge 1
                        start_time = time.time()
                        rouge1_result = rouge1.score(target=gold_summary, prediction=candidate_summary)["rouge1"][2]  # 0: precision, 1: recall, 2: fmeasure
                        duration = time.time() - start_time
                        append_score(self, source_file=source_file, type="N-gram", method="Rouge1", candidate_variant=candidate_variant, result=rouge1_result, duration=duration)

                        # Rouge 2
                        start_time = time.time()
                        rouge2_result = rouge2.score(target=gold_summary, prediction=candidate_summary)["rouge2"][2]
                        duration = time.time() - start_time
                        append_score(self, source_file=source_file, type="N-gram", method="Rouge2", candidate_variant=candidate_variant, result=rouge2_result, duration=duration)

                        # ---------GRAPH
                        start_time = time.time()
                        autosummeng_score, memog_score, npower_score = npower.score(target=source_path, prediction=candidate_path)
                        duration = time.time() - start_time

                        # AutoSummENG
                        append_score(self, source_file=source_file, type="Graph-based", method="AutoSummENG", candidate_variant=candidate_variant, result=autosummeng_score, duration=duration)

                        # MeMoG
                        append_score(self, source_file=source_file, type="Graph-based", method="MeMoG", candidate_variant=candidate_variant, result=memog_score, duration=duration)

                        # ---------PROBABILISTIC
                        # SummTriver

                        # ---------META
                        # NPowER - computed in graph methods
                        append_score(self, source_file=source_file, type="Meta", method="NPowER", candidate_variant=candidate_variant, result=npower_score, duration=duration)

                        # ---------EMBEDDINGS-BASED
                        # BERTScore
                        start_time = time.time()
                        P, R, F1 = bertscore.score([candidate_summary], [gold_summary])
                        duration = time.time() - start_time
                        append_score(self, source_file=source_file, type="Embeddings-based", method="BERTScore", candidate_variant=candidate_variant, result=float(F1), duration=duration)

                        if LANGUAGE_CODE == "en":

                            # BARTscore
                            start_time = time.time()
                            bartscore_result = bartscore.score([candidate_summary], [gold_summary], batch_size=4)
                            duration = time.time() - start_time
                            append_score(self, source_file=source_file, type="Embeddings-based", method="BARTScore", candidate_variant=candidate_variant, result=bartscore_result[0], duration=duration)

                            # Bleurt
                            start_time = time.time()
                            bleurt_result = bleurt.score(references=[gold_summary], candidates=[candidate_summary])
                            duration = time.time() - start_time
                            append_score(self, source_file=source_file, type="Embeddings-based", method="Bleurt", candidate_variant=candidate_variant, result=bleurt_result[0], duration=duration)

            except FileNotFoundError as e:
                logger.error(f"File not found: {e}. Skipping {source_file}.")
                continue
            results = pd.DataFrame.from_dict(self.data, orient="index").transpose()
            results.to_csv(RESULTS_PATH, index=False)
        logger.info("Summary evaluation completed.")
        return results
