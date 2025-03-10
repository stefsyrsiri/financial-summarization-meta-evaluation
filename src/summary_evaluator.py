import os
import subprocess
import pandas as pd
from loguru import logger
from rouge_score.rouge_scorer import RougeScorer
# from transformers import BertTokenizer, BertForMaskedLM, BertModel
from bert_score import BERTScorer


class SummaryEvaluator:
    def __init__(self, gold_dir: str, candidate_dir: str):
        self.gold_dir = gold_dir
        self.candidate_dir = candidate_dir
        self.data = {
            'source_doc': [],
            'eval_type': [],
            'eval_method': [],
            'variant': [],
            'score': [],
        }
        self.rouge1 = RougeScorer(['rouge1'], use_stemmer=True)
        self.rouge2 = RougeScorer(['rouge2'], use_stemmer=True)
        self.bert_scorer = BERTScorer(lang='el')
        logger.info("SummaryEvaluator initialized.")

    def evaluate_summaries(self, source_docs):
        logger.info("Starting summary evaluation.")
        for source_file in source_docs:
            source_path = os.path.join(self.gold_dir, f'{source_file}_1.txt')
            with open(source_path, mode='r', encoding='utf-8') as gold_f:
                gold_summary = gold_f.read()

            candidate_summaries = [doc for doc in os.listdir(self.candidate_dir) if doc.startswith(f'{source_file}_')]
            other_summaries = [doc for doc in os.listdir(self.gold_dir) if not doc.startswith(f'{source_file}_') and doc.endswith('_1.txt')]
            candidate_summaries.extend(other_summaries[:10])
            candidate_summaries.insert(0, source_file)

            for candidate_file in candidate_summaries:
                logger.info(f"Evaluating candidate summary: {candidate_file}")
                # Source
                if candidate_file == source_file:
                    candidate_path = os.path.join(self.gold_dir, f'{candidate_file}_1.txt')
                    candidate_variant = 'source'
                # Other (gold) summaries
                elif candidate_file.endswith('_1.txt'):
                    candidate_path = os.path.join(self.gold_dir, candidate_file)
                    candidate_variant = candidate_file.removesuffix('.txt')
                # Candidate / Destroyed summaries
                else:
                    candidate_path = os.path.join(self.candidate_dir, candidate_file)
                    candidate_variant = candidate_file.removeprefix(f'{source_file}_').removesuffix('.txt')

                with open(candidate_path, mode='r', encoding='utf-8') as cand_f:
                    candidate_summary = cand_f.read()

                    # ----------N-GRAM
                    # Rouge 1
                    rouge1_result = self.rouge1.score(target=gold_summary, prediction=candidate_summary)['rouge1'][2]  # fmeasure
                    self._append_score(source_file=source_file, type='N-gram', method='Rouge1', candidate_variant=candidate_variant, result=rouge1_result)

                    # Rouge 2
                    rouge2_result = self.rouge2.score(target=gold_summary, prediction=candidate_summary)['rouge2'][2]  # fmeasure
                    self._append_score(source_file=source_file, type='N-gram', method='Rouge2', candidate_variant=candidate_variant, result=rouge2_result)

                    # ---------GRAPH
                    autosummeng_score, memog_score, npower_score = self.npower(target=source_path, prediction=candidate_path)

                    # AutoSummENG
                    self._append_score(source_file=source_file, type='Graph-based', method='AutoSummENG', candidate_variant=candidate_variant, result=autosummeng_score)

                    # MeMoG
                    self._append_score(source_file=source_file, type='Graph-based', method='MeMoG', candidate_variant=candidate_variant, result=memog_score)

                    # ---------PROBABILISTIC
                    # SummTriver

                    # ---------META
                    # NPowER - computed in graph methods
                    self._append_score(source_file=source_file, type='Meta', method='NPowER', candidate_variant=candidate_variant, result=npower_score)

                    # FRESA

                    # BRUGEscore

                    # ---------EMBEDDINGS-BASED
                    # BERTScore
                    P, R, F1 = self.bert_scorer.score([candidate_summary], [gold_summary])
                    self._append_score(source_file=source_file, type='Embeddings-based', method='BERTScore', candidate_variant=candidate_variant, result=float(F1))

                    # BARTscore

                    # Bleurt

                    # ---------TRANSFORMER-BASED
                    # GPTscore

                    # G-Eval

                    # Extract-then-Evaluate

        logger.info("Summary evaluation completed.")
        return pd.DataFrame.from_dict(self.data, orient='index').transpose()

    def _append_score(self, source_file, type, method, candidate_variant, result):
        self.data['source_doc'].append(source_file)
        self.data['eval_type'].append(type)
        self.data['eval_method'].append(method)
        self.data['variant'].append(candidate_variant)
        self.data['score'].append(result)

    def npower(self, target, prediction, min_n=3, max_n=3, dwin=3, min_score=0.0, max_score=1.0):
        result = subprocess.check_output([
            "java",
            "-jar",
            "NPowERV1/V1/NPowER.jar",
            f"-peer={target}",
            f"-models={prediction}",
            f"[-minN={min_n}]",
            f"[-maxN={max_n}]",
            f"[-dwin={dwin}]",
            f"[-minScore={min_score}]",
            f"[-maxScore={max_score}]",
            "-allScores",
            "[-noSelfModel]"
            ], text=True, stderr=subprocess.DEVNULL)
        return result.strip().split()
