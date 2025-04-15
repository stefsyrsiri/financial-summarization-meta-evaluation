import os
import torch
import pandas as pd
from loguru import logger
from rouge_score.rouge_scorer import RougeScorer
from evaluation_methods.BARTScore.bart_score import BARTScorer
from bert_score import BERTScorer
from evaluation_methods.Bleurt.bleurt.score import BleurtScorer
from evaluation_methods.NPowERV1.npower_score import NPowERScorer

SUMMARY_VER = os.getenv('SUMMARY_VER')
FILE_EXTENSION = os.getenv('FILE_EXTENSION')

rouge1 = RougeScorer(['rouge1'], use_stemmer=True)
rouge2 = RougeScorer(['rouge2'], use_stemmer=True)
bertscore = BERTScorer(lang='el')
npower = NPowERScorer()
bartscore = BARTScorer(device='cuda:0' if torch.cuda.is_available() else 'cpu', checkpoint='facebook/bart-large-cnn')  # around 2 mins to load
checkpoint = "evaluation_methods/Bleurt/bleurt/test_checkpoint"
bleurt = BleurtScorer(checkpoint)


class SummaryEvaluator:
    def __init__(
            self,
            gold_dir: str,
            candidate_dir: str,
            rouge1: RougeScorer = rouge1,
            rouge2: RougeScorer = rouge2,
            npower: NPowERScorer = npower,
            bertscore: BERTScorer = bertscore,
            bartscore: BARTScorer = bartscore,
            bleurt: BleurtScorer = bleurt
            ):
        self.gold_dir = gold_dir
        self.candidate_dir = candidate_dir
        self.data = {
            'source_doc': [],
            'eval_type': [],
            'eval_method': [],
            'variant': [],
            'score': [],
        }
        self.rouge1 = rouge1
        self.rouge2 = rouge2
        self.npower = npower
        self.bertscore = bertscore
        self.bartscore = bartscore
        self.bleurt = bleurt
        logger.info("SummaryEvaluator initialized.")

    def evaluate_summaries(self, source_docs):
        logger.info("Starting summary evaluation.")
        for source_file in source_docs:
            source_path = os.path.join(self.gold_dir, f'{source_file}{SUMMARY_VER}{FILE_EXTENSION}')
            with open(source_path, mode='r', encoding='utf-8') as gold_f:
                gold_summary = gold_f.read()

            candidate_summaries = [doc for doc in os.listdir(self.candidate_dir) if doc.startswith(f'{source_file}_')]
            other_summaries = [doc for doc in os.listdir(self.gold_dir) if not doc.startswith(f'{source_file}_') and doc.endswith(f'{SUMMARY_VER}{FILE_EXTENSION}')]
            candidate_summaries.extend(other_summaries[:10])
            candidate_summaries.insert(0, source_file)

            for candidate_file in candidate_summaries:
                logger.info(f"Evaluating candidate summary: {candidate_file}")
                # Source
                if candidate_file == source_file:
                    candidate_path = os.path.join(self.gold_dir, f'{candidate_file}{SUMMARY_VER}{FILE_EXTENSION}')
                    candidate_variant = 'source'
                # Other (gold) summaries
                elif candidate_file.endswith(f'{SUMMARY_VER}{FILE_EXTENSION}'):
                    candidate_path = os.path.join(self.gold_dir, candidate_file)
                    candidate_variant = candidate_file.removesuffix('{FILE_EXTENSION}')
                # Candidate / Destroyed summaries
                else:
                    candidate_path = os.path.join(self.candidate_dir, candidate_file)
                    candidate_variant = candidate_file.removeprefix(f'{source_file}_').removesuffix('{FILE_EXTENSION}')

                with open(candidate_path, mode='r', encoding='utf-8') as cand_f:
                    candidate_summary = cand_f.read()

                    # ----------N-GRAM
                    # Rouge 1
                    rouge1_result = self.rouge1.score(target=gold_summary, prediction=candidate_summary)['rouge1'][1]  # 0: precision, 1: recall, 2: fmeasure
                    self._append_score(source_file=source_file, type='N-gram', method='Rouge1', candidate_variant=candidate_variant, result=rouge1_result)

                    # Rouge 2
                    rouge2_result = self.rouge2.score(target=gold_summary, prediction=candidate_summary)['rouge2'][1]
                    self._append_score(source_file=source_file, type='N-gram', method='Rouge2', candidate_variant=candidate_variant, result=rouge2_result)

                    # ---------GRAPH
                    autosummeng_score, memog_score, npower_score = self.npower.score(target=source_path, prediction=candidate_path)

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
                    P, R, F1 = self.bertscore.score([candidate_summary], [gold_summary])
                    self._append_score(source_file=source_file, type='Embeddings-based', method='BERTScore', candidate_variant=candidate_variant, result=float(F1))

                    # BARTscore
                    bartscore_result = self.bartscore.score([candidate_summary], [gold_summary], batch_size=4)
                    self._append_score(source_file=source_file, type='Embeddings-based', method='BARTScore', candidate_variant=candidate_variant, result=bartscore_result[0])

                    # Bleurt
                    bleurt_result = self.bleurt.score(references=[gold_summary], candidates=[candidate_summary])
                    self._append_score(source_file=source_file, type='Embeddings-based', method='Bleurt', candidate_variant=candidate_variant, result=bleurt_result[0])

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
