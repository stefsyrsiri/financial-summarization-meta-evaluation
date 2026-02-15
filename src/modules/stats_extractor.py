import os

import pandas as pd
from tqdm.auto import tqdm
from loguru import logger

from transformers import BertTokenizer

from src.modules.tokenizer import Tokenizer
from src.registries.languages_registry import LANGUAGES

tqdm.pandas()


class StatsExtractor:
    """Class to extract text statistics from the dataset, including spaCy and BERT token counts."""
    def __init__(
            self,
            dataset_path: str,
            spacy_tokenizer: Tokenizer,
            bert_tokenizer: BertTokenizer,
            results_path: str="results/eda/text_stats.parquet"
            ):
        self.dataset_path = dataset_path
        self.spacy_tokenizer = spacy_tokenizer
        self.bert_tokenizer = bert_tokenizer
        self.results_path = results_path

    @staticmethod
    def get_dataset(dataset_path: str) -> None:
        """Creates a unified dataset from all the .txt files with all their directory metadata."""
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


    def _filter_dataset(self, df: pd.DataFrame):
        """Filters the dataset to only include the documents that have used in our experiments.

        Args:
            df (pd.DataFrame): The full dataset with all documents and metadata.

        Returns:
            pd.DataFrame: The filtered dataset with only the relevant documents.
        """
        logger.info(f"Filtering dataset with size: {len(df)}.")
        # For English, only keep the documents that were sampled
        with open("results/sampling/English_sampled_docs.txt", "r") as f:
            sampled_docs = f.read().splitlines()
        df = df[~((df['language'] == "English") & (~df['doc_id'].isin(sampled_docs)))]

        # Filter by language and gold summary version (e.g. for English we only care about `_1` summaries)
        en_gold = df[(df['language'] == 'English') & (df['doc_type'] == 'gold_summaries') & (df['version'] == '1')]
        el_gold = df[(df['language'] == 'Greek') & (df['doc_type'] == 'gold_summaries') & (df['version'] == '2')]
        es_gold = df[(df['language'] == 'Spanish') & (df['doc_type'] == 'gold_summaries') & (df['version'] == 'GS1')]

        # Remove all gold summaries from the df, as the ones we care about will be concatenated back in.
        df_no_gold = df[df['doc_type'] != 'gold_summaries']

        # Remove all candidate_summaries (this leaves us with the candidate_summaries_trunc)
        df_no_gold_no_cand = df_no_gold[df_no_gold['doc_type'] != 'candidate_summaries']

        # Final: annual reports, gold summaries (filtered by version), candidate summaries truncated (filtered at creation time)
        final_df = pd.concat([df_no_gold_no_cand, en_gold, el_gold, es_gold])
        logger.info(f"Final filtered dataset size: {len(final_df)}")
        return final_df


    def _compute_spacy_stats(
            self,
            df: pd.DataFrame,
            batch_size: int=1,
            n_process: int=1
            ):
        """Computes spaCy token and sentence counts for each document in the dataset, grouped by language and doc type.

        Args:
            df (pd.DataFrame): The filtered dataset containing the documents and metadata.
            batch_size (int, optional): Batch size for spaCy processing. Defaults to 1.
            n_process (int, optional): Number of processes for spaCy. Defaults to 1.

        Returns:
            pd.DataFrame: A dataframe with the spaCy token and sentence counts for each document.
        """
        logger.info("Computing spaCy text statistics...")
        for language, code in LANGUAGES.items():
            # ONLY for gold_summaries
            spacy_filter = (df['language'] == language) & (df['doc_type'] == 'gold_summaries')
            self.spacy_tokenizer.lang_code = code
            spacy_texts = df.loc[spacy_filter, 'text'].to_list()

            spacy_token_counts = []
            sentence_counts = []

            try:
                for doc in self.spacy_tokenizer.nlp.pipe(
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
        return df


    def _compute_bert_stats(self, df: pd.DataFrame):
        """Computes BERT token counts for each document in the dataset, grouped by language.

        Args:
            df (pd.DataFrame): The filtered dataset containing the documents and metadata.

        Returns:
            pd.DataFrame: A dataframe with the BERT token counts for each document.
        """
        logger.info("Computing BERT text statistics...")
        for language in LANGUAGES.keys():
            bert_filter = df['language'] == language
            texts = df.loc[bert_filter, 'text'].to_list()

            try:
                bert_token_counts = []
                for text in tqdm(texts, desc=f"BERT: Processing {language}", unit="doc"):
                    tokens = self.bert_tokenizer.encode(text, add_special_tokens=False)
                    bert_token_counts.append(len(tokens))

                df.loc[bert_filter, 'bert_token_count'] = bert_token_counts
            except Exception as e:
                logger.error(f"Error processing BERT stats for {language}: {e}")
        return df


    def get_stats(self, df: pd.DataFrame) -> None:
        """Main function to compute text statistics for the dataset, including filtering and saving results.

        Args:
            df (pd.DataFrame): The full dataset with all documents and metadata.
        """
        logger.info("Calculating text statistics...")

        # Filter dataset
        df = self._filter_dataset(df)

        # Compute spaCy stats
        df_spacy = self._compute_spacy_stats(df)
        df_bert = self._compute_bert_stats(df)

        # Merge spaCy and BERT stats back into the main dataframe
        df_spacy_merged = df.merge(df_spacy, on=list(df.columns), how='left')
        df_bert_merged = df_spacy_merged.merge(df_bert, on=list(df_spacy_merged.columns), how='left')
        df_bert_merged

        # Save final results
        df_bert_merged.to_parquet(self.results_path)
        logger.info(f"All text statistics saved to {self.results_path}")
