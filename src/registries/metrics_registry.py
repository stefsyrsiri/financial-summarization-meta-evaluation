from typing import Literal
from dataclasses import dataclass

import torch
from longdocfactscore.ldfacts import LongDocFACTScore
from rouge_score.rouge_scorer import RougeScorer
from bert_score import BERTScorer

from evaluation_methods.BARTScore.bart_score import BARTScorer
from evaluation_methods.Bleurt.bleurt.score import BleurtScorer
from evaluation_methods.NPowERV1 import npower
from evaluation_methods.FactCC.factcc import batched_FactCC
from evaluation_methods.FinSumEval.metric.evaluators.finsumeval import SumEval
from src.modules.tokenizer import Tokenizer

device = "cuda:0" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class EvalMetric:
    name: str
    eval_type: Literal[
        "N-gram-based",
        "N-gram-graph-based",
        "Embeddings-based",
        "NLI",
        "Meta",
    ]
    device: Literal["cpu", "cuda"]
    multilingual: bool
    needs_ref: bool


METRICS = {
    "rouge1": EvalMetric("ROUGE-1", "N-gram-based", "cpu", True, True),
    "rouge2": EvalMetric("ROUGE-2", "N-gram-based", "cpu", True, True),
    "autosummeng": EvalMetric("AutoSummENG", "N-gram-graph-based", "cpu", True, True),
    "memog": EvalMetric("MeMoG", "N-gram-graph-based", "cpu", True, True),
    "npower": EvalMetric("NPowER", "N-gram-graph-based", "cpu", True, True),
    "bertscore": EvalMetric("BERTScore", "Embeddings-based", "cuda", True, True),
    "bartscore": EvalMetric("BARTScore", "Embeddings-based", "cuda", False, True),
    "ldfactscore": EvalMetric("LongDocFACTScore", "Embeddings-based", "cuda", False, False),
    "bleurt": EvalMetric("BLEURT", "NLI", "cuda", True, True),
    "factcc": EvalMetric("FactCC", "NLI", "cuda", False, True),
    "finsumeval": EvalMetric("FinSumEval", "Meta", "cuda", True, False),
}


METRIC_FACTORIES = {
    "rouge1": lambda lang: RougeScorer(["rouge1"], use_stemmer=False, tokenizer=Tokenizer(lang)),
    "rouge2": lambda lang: RougeScorer(["rouge2"], use_stemmer=False, tokenizer=Tokenizer(lang)),
    "autosummeng": lambda lang: npower,
    "memog": lambda lang: npower,
    "npower": lambda lang: npower,
    "bertscore": lambda lang: BERTScorer(lang=lang, device=device),
    "bartscore": lambda lang: BARTScorer(device=device, checkpoint="facebook/bart-large-cnn"),
    "bleurt": lambda lang: BleurtScorer(checkpoint="evaluation_methods/Bleurt/bleurt/BLEURT-20"),
    "factcc": lambda lang: batched_FactCC,
    "ldfactscore": lambda lang: LongDocFACTScore(device=device),
    "finsumeval": lambda lang: SumEval(lang=lang),
}
