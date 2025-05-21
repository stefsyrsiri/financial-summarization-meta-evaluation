"""
This module provides tools that destroy summaries by:

- swapping their words
- removing words
- removing sentences
- inserting sentences from other summaries (from the same dataset)
- repeating sentences
"""

import math
import os
from random import choice, sample  # sampling without replacement

from dotenv import load_dotenv
from loguru import logger

from utils.summary_corruptor_utils import get_swap_indices
from modules.tokenizer import Tokenizer

load_dotenv(override=True)
SUMMARY_VER = os.getenv("SUMMARY_VER")
FILE_EXTENSION = os.getenv("FILE_EXTENSION")
LANGUAGE_CODE = os.getenv("LANGUAGE_CODE")


class SummaryCorruptor:
    def __init__(
            self,
            input_summary: str,
            noise_percentage: float,
            language: str = LANGUAGE_CODE,
            ):
        """Makes summaries noisy.

        Args:
            input_summary (str): The string of the summary to be destroyed.
            noise_percentage (float): The percentage of the summary to be modified.
            language (str): The language of the summary. Default is dotenv variable "LANGUAGE_CODE".
                            The language should be passed as an ISO 639-1 code.
                            Supported languages: "en" (English), "el" (Greek), "es" (Spanish).

        """
        if not (0 < noise_percentage < 1):
            raise ValueError("summary_perc must be positive and less than 1.")
        self.input_summary = input_summary
        self.nlp = Tokenizer(lang_code=language)
        self.words = self.nlp.tokenize(self.input_summary)
        self.word_indices = [index for index in range(len(self.words) - 1)]
        self.sentences = self.nlp.sentencize(self.input_summary)
        self._noise_percentage = noise_percentage
        self._random_swap_word_indices = None
        self._consecutive_swap_word_indices = None
        self._removed_words = None
        self._removed_sentences = None
        self._random_summary = None
        self._random_sentences = None
        self._repeated_sentence = None

        logger.info(f"SummaryCorruptor initialized with {self.noise_percentage:.0%} noise.")
        logger.debug(f"Tokens: {len(self.words)}")
        logger.debug(f"Sentences: {len(self.sentences)}")

    @property
    def noise_percentage(self):
        return self._noise_percentage

    @noise_percentage.setter
    def noise_percentage(self, new_percentage: float):
        if isinstance(new_percentage, float) and 0 < new_percentage < 1:
            self._noise_percentage = new_percentage
            logger.info(f"SummaryCorruptor's noise percentage was changed to {self.noise_percentage:.0%}.")

    @property
    def random_swap_word_indices(self):
        return self._random_swap_word_indices

    @random_swap_word_indices.setter
    def random_swap_word_indices(self, new_random_swap_word_indices: list[int]):
        if isinstance(new_random_swap_word_indices, list):
            self._random_swap_word_indices = new_random_swap_word_indices

    @property
    def consecutive_swap_word_indices(self):
        return self._consecutive_swap_word_indices

    @consecutive_swap_word_indices.setter
    def consecutive_swap_word_indices(self, new_consecutive_swap_word_indices: list[int]):
        if isinstance(new_consecutive_swap_word_indices, list):
            self._consecutive_swap_word_indices = new_consecutive_swap_word_indices

    @property
    def removed_words(self):
        return self._removed_words

    @removed_words.setter
    def removed_words(self, new_removed_words: list[int]):
        if isinstance(new_removed_words, list):
            self._removed_words = new_removed_words

    @property
    def removed_sentences(self):
        return self._removed_sentences

    @removed_sentences.setter
    def removed_sentences(self, new_removed_sentences: list[str]):
        if isinstance(new_removed_sentences, list):
            self._removed_sentences = new_removed_sentences

    @property
    def random_summary(self):
        return self._random_summary

    @random_summary.setter
    def random_summary(self, new_random_summary: str):
        if isinstance(new_random_summary, str):
            self._random_summary = new_random_summary

    @property
    def random_sentences(self):
        return self._random_sentences

    @random_sentences.setter
    def random_sentences(self, new_random_sentences: list[str]):
        if isinstance(new_random_sentences, list):
            self._random_sentences = new_random_sentences

    @property
    def repeated_sentence(self):
        return self._repeated_sentence

    @repeated_sentence.setter
    def repeated_sentence(self, new_repeated_sentence: str):
        if isinstance(new_repeated_sentence, str):
            self._repeated_sentence = new_repeated_sentence

    def random_swap_words(self) -> str:
        """Swaps words from the summary n_swap times.

        Returns:
            str: The summary with the swapped words.

        """
        split_summary = self.words.copy()
        # Get summary indices and target number for swapping
        target_number = get_swap_indices(self)

        # Sample n_words2swap number of words indices
        if self.random_swap_word_indices is None:
            word_indices = sample(self.word_indices, k=target_number * 2)
        else:
            word_indices = sample(self.random_swap_word_indices, k=target_number * 2)
        self.random_swap_word_indices = word_indices
        logger.debug(f"Number of indices to be swapped: {target_number * 2}")
        logger.debug(f"Indices to be swapped: {word_indices}")

        # Swap the words and slide indices from left to the right
        word_indices_len = len(word_indices)
        index1, index2 = 0, 1

        try:
            while index2 < word_indices_len:
                logger.debug(f"Indices: {word_indices[index1]}, {word_indices[index2]} = {word_indices[index2]}, {word_indices[index1]}")
                logger.debug(f"Words: {split_summary[word_indices[index1]]}, {split_summary[word_indices[index2]]} = {split_summary[word_indices[index2]]}, {split_summary[word_indices[index1]]}")
                split_summary[word_indices[index1]], split_summary[word_indices[index2]] = split_summary[word_indices[index2]], split_summary[word_indices[index1]]
                index1 += 2
                index2 += 2
        except IndexError:
            print(str(index1)+" "+str(index2)+" "+str(word_indices_len))
        return " ".join(split_summary)

    def consecutive_swap_words(self) -> str:
        """Consecutively swaps words from the summary n_swap times.

        Returns:
            str: The summary with the swapped words.

        """
        split_summary = self.words.copy()

        # Get summary indices and target number for swapping
        target_number = get_swap_indices(self)

        # Sample n_swaps number of words indices
        if self.consecutive_swap_word_indices is None:
            word_indices = sample(self.word_indices[:-1], k=target_number)
        else:
            word_indices = sample(self.consecutive_swap_word_indices[:-1], k=target_number)
        self.consecutive_swap_word_indices = word_indices
        word_indices.sort()
        logger.debug(f"Number of indices to be swapped: {target_number * 2}")
        logger.debug(f"Indices to be swapped: {word_indices}")

        # Swap the words and slide indices from left to the right
        already_swapped = []
        for index in word_indices:
            if index not in already_swapped:
                already_swapped.extend([index, index + 1])
                logger.debug(f"Indices: {index}, {index + 1} = {index + 1}, {index}")
                logger.debug(f"Words: {split_summary[index]}, {split_summary[index + 1]} = {split_summary[index + 1]}, {split_summary[index]}")
                split_summary[index], split_summary[index + 1] = split_summary[index + 1], split_summary[index]
        logger.debug(f"Number of words swapped: {len(already_swapped)}")
        logger.debug(f"Indices swapped: {already_swapped}")
        return " ".join(split_summary)

    def remove_words(self) -> str:
        """Removes n_words from the summary.

        Returns:
            str: The summary with the deleted words.

        """
        split_summary = self.words.copy()
        summary_len = len(split_summary)
        logger.debug(f"Total words: {summary_len}")

        target_number = math.ceil(summary_len * self.noise_percentage)
        logger.debug(f"Number of words to be removed: {target_number}")

        logger.debug("Words removed:")
        if self.removed_words is None:
            random_words = sample(split_summary, k=target_number)
        else:
            random_words = sample(self.removed_words, k=target_number)
        self.removed_words = random_words
        for i, random_word in enumerate(random_words):
            logger.debug(f"{i}: {random_word}")
            split_summary.remove(random_word)
        return " ".join(split_summary)

    def remove_sentence(self) -> str:
        """Removes a sentence from the summary.

        Returns:
            str: The summary after the removal of the sentence.

        """
        summary_len = len(self.sentences)
        logger.debug(f"Total sentences: {summary_len}")

        target_number = math.ceil(summary_len * self.noise_percentage)
        logger.debug(f"Number of sentences to be removed: {target_number}")

        # Remove sentences
        if self.removed_sentences is None:
            sents_remove = sample(self.sentences, k=target_number)
        else:
            sents_remove = sample(self.removed_sentences, k=target_number)
        self.removed_sentences = sents_remove
        logger.debug("Sentences to be removed:")
        for i, sent in enumerate(sents_remove):
            logger.debug(f"{i}:\n{sent}\n")
        new_text = [sent for sent in self.sentences if sent not in sents_remove]
        return "".join(new_text)

    def insert_sentence(
            self,
            target: str,
            source_docs: list,
            gold_dir: str
            ) -> str:
        """Inserts a sentence from another summary in the dataset.

        Args:
            target (str): The number of the summary.
            source_docs (list): The list of annual reports in the analysis.
            gold_dir (str): The path of the gold summaries.

        Returns:
            str: The summary with the new inserted sentence.

        """
        summary_len = len(self.sentences)
        logger.debug(f"Total sentences: {summary_len}")

        # Choose another summary
        if self.random_summary is None:
            remaining_summaries = [doc for doc in source_docs if doc != target]
            rand_summary = choice(remaining_summaries)
        else:
            rand_summary = self.random_summary
        logger.debug(f"Random summary: {rand_summary}")

        # Choose a sentence from the randomly picked summary to insert to the original one
        with open(os.path.join(gold_dir, f"{rand_summary}{SUMMARY_VER}{FILE_EXTENSION}"), mode="r", encoding="utf-8") as file:
            rand_gold_summary = file.read()
            rand_sentences = self.nlp.sentencize(rand_gold_summary)
            rand_sentence_len = len(rand_sentences)
            logger.debug(f"Random summary sentences: {rand_sentence_len}")

            target_number = math.ceil(summary_len * self.noise_percentage)
            logger.debug(f"Number of sentences to insert: {target_number}")

            # Insert new sentences
            if self.random_sentences is None:
                if target_number > rand_sentence_len:
                    target_number = rand_sentence_len
                sents_insert = sample(rand_sentences, k=target_number)
            else:
                if target_number > len(self.random_sentences):
                    target_number = len(self.random_sentences)
                sents_insert = sample(self.random_sentences, k=target_number)
            self.random_sentences = sents_insert
            logger.debug("Sentences to be inserted:")
            for i, sent in enumerate(sents_insert):
                logger.debug(f"{i}:\n{sent}\n")
            original_text = "".join([sent for sent in self.sentences])
            new_text = "".join(sents_insert)
            return new_text + original_text

    def repeat_sentence(self) -> str:
        """Repeats a sentence in the summary.

        Returns:
            str: The summary with the repeated sentences.

        """
        summary_len = len(self.sentences)
        logger.debug(f"Total sentences: {summary_len}")

        target_number = math.ceil(summary_len * self.noise_percentage)
        logger.debug(f"Number of times to repeat: {target_number}")

        if self.repeated_sentence is None:
            self.repeated_sentence = choice(self.sentences)
        logger.debug(f"Sentence to be repeated: {self.repeated_sentence}")
        original_text = "".join([sent for sent in self.sentences])
        new_text = "".join([self.repeated_sentence] * target_number)
        return new_text + original_text
