import os

import pandas as pd
import spacy
from tqdm.auto import tqdm
from loguru import logger

from transformers import BertTokenizer

from src.modules.tokenizer import Tokenizer
from src.registries.languages_registry import LANGUAGES

tqdm.pandas()


def get_dataset(dataset_path: str):
    """Creates a unified dataset from all the .txt files with all their directory metadata.
    
    Args:
        dataset_path (str): Path to save/load the unified dataset (parquet file).
    """
    logger.info(f"Creating/loading dataset at {dataset_path}...")
    if os.path.exists(dataset_path):
        return pd.read_parquet(dataset_path)
    else:
        rows = []

        # Every language (English, Greek, Spanish)
        for language in tqdm(os.listdir('data/'), desc="Processing languages"):
            logger.info(f"Processing language: {language}")

            # Training dataset only (exclude validation)
            for dataset in ['training']:

            # Every doc type (annual reports, gold summaries, candidate summaries, candidate summaries truncated)
                for doc_type in ['annual_reports', 'gold_summaries', 'candidate_summaries', 'candidate_summaries_trunc']:
                    logger.info(f"Processing doc type: {doc_type}")

                    folder_path = f'data/{language}/{dataset}/{doc_type}'
                    if not os.path.exists(folder_path):
                        continue
                    for file in os.listdir(folder_path):
                        with open(f'{folder_path}/{file}', 'r', encoding='utf-8') as f:
                            text = f.read()
                            file_name = file.removesuffix(os.getenv('FILE_EXTENSION', '.txt'))

                            # Extract directory metadata
                            if doc_type == 'annual_reports':
                                doc_id = file_name
                                summary_version = None
                                noise_variant = None
                            else:
                                try:
                                    parts = file_name.split("_")
                                    doc_id = parts[0]
                                    summary_version = parts[-1]
                                    noise_variant = None
                                    if doc_type in ['candidate_summaries', 'candidate_summaries_trunc']:
                                        noise_variant = "_".join(parts[1:-1])
                                except Exception as e:
                                    logger.error(f"Error processing {file}: {e}")
                                    logger.debug(f"File name: {file_name}")
                                    logger.debug(f"Doc ID: {doc_id}")
                            rows.append({
                                'doc_id': doc_id,
                                'dataset': dataset,
                                'version': summary_version,
                                'noise_variant': noise_variant,
                                'doc_type': doc_type,
                                'language': language,
                                'text': text,
                            })

        df = pd.DataFrame(rows)
        df.to_parquet(dataset_path)


def _filter_dataset(df: pd.DataFrame):
    en_gold = df[(df['language'] == 'English') & (df['doc_type'] == 'gold_summaries') & (df['version'] == '1')]
    el_gold = df[(df['language'] == 'Greek') & (df['doc_type'] == 'gold_summaries') & (df['version'] == '2')]
    es_gold = df[(df['language'] == 'Spanish') & (df['doc_type'] == 'gold_summaries') & (df['version'] == 'GS1')]
    no_gold = df[df['doc_type'] != 'gold_summaries']
    no_cand = no_gold[no_gold['doc_type'] != 'candidate_summaries']
    return pd.concat([no_cand, en_gold, el_gold, es_gold])


def _compute_spacy_stats(
        df: pd.DataFrame,
        spacy_tokenizer: Tokenizer,
        batch_size: int=1,
        n_process: int=1
        ):
    logger.info("Computing spaCy text statistics...")
    for language, code in LANGUAGES.items():
        # ONLY for candidate_summaries_trunc
        spacy_filter = (df['language'] == language) & (df['doc_type'] == 'candidate_summaries_trunc')
        spacy_tokenizer.lang_code = code
        spacy_texts = df.loc[spacy_filter, 'text'].to_list()

        spacy_token_counts = []
        sentence_counts = []

        try:
            for doc in spacy_tokenizer.nlp.pipe(
                tqdm(spacy_texts, desc=f"spaCy: Processing {language}", unit="doc"),
                batch_size=batch_size,
                n_process=n_process
                ):
                spacy_token_counts.append(len(doc))
                sentence_counts.append(len([sent.text for sent in doc.sents]))

            df.loc[spacy_filter, 'spacy_token_count'] = spacy_token_counts
            df.loc[spacy_filter, 'spacy_sent_count'] = sentence_counts
        except Exception as e:
            logger.error(f"Error processing spaCy stats for {language}: {e}")
    df.to_parquet("temp_spacy_stats.parquet")


def _compute_bert_stats(df: pd.DataFrame, bert_tokenizer: BertTokenizer):
    logger.info("Computing BERT text statistics...")
    for language in LANGUAGES.keys():
        bert_filter = df['language'] == language
        texts = df.loc[bert_filter, 'text'].to_list()

        try:
            bert_token_counts = []
            for text in tqdm(texts, desc=f"BERT: Processing {language}", unit="doc"):
                tokens = bert_tokenizer.encode(text, add_special_tokens=False)
                bert_token_counts.append(len(tokens))

            df.loc[bert_filter, 'bert_token_count'] = bert_token_counts
        except Exception as e:
            logger.error(f"Error processing BERT stats for {language}: {e}")
    df.to_parquet("temp_bert_stats.parquet")


def get_stats(
        df: pd.DataFrame,
        spacy_tokenizer: Tokenizer,
        bert_tokenizer: BertTokenizer,
        results_path: str="text_stats.parquet"
        ):
    logger.info("Calculating text statistics...")

    # Filter dataset
    df = _filter_dataset(df)

    # Compute spaCy stats
    _compute_spacy_stats(df, spacy_tokenizer)
    _compute_bert_stats(df, bert_tokenizer)

    # Save final results
    df_spacy = pd.read_parquet("temp_spacy_stats.parquet")
    df_bert = pd.read_parquet("temp_bert_stats.parquet")

    # df.to_parquet(results_path)
    # logger.info(f"All text statistics saved to {results_path}")
