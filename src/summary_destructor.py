"""
This module provides tools that destroy summaries by:

- swapping their words
- removing words
- removing sentences
- inserting sentences from other summaries (from the same dataset)
- repeating sentences
"""


import os
from loguru import logger
import spacy
from random import choice, choices, sample
from typing import Optional

nlp = spacy.load('el_core_news_sm')
nlp.add_pipe('sentencizer', before='parser')


class SummaryDestructor:
    def __init__(
            self,
            input_summary: str,
            ):
        """Makes summaries noisy.

        Args:
            input_summary (str): The string of the summary to be destroyed.

        """
        self.input_summary = input_summary
        logger.info("SummaryDestructor initialized.")

    # Helper for the random_swap and consecutive_swap methods
    def _get_swap_indices(
            self,
            n_swaps: Optional[int] = None,
            summary_perc: Optional[float] = None
            ):
        """Helper method for the random_swap_words and consecutive_swap_words methods.
           Prepares the summary indices and target number used for swapping.

        Args:
            n_swaps (Optional[int]): The number of swaps to occur.
            summary_perc (Optional[float]): The percentage of the summary to be modified.

        Returns:
            The summary indices and the target number used for sampling summary indices.

        """
        split_summary = self.input_summary.split()
        summary_len = len(split_summary)
        logger.debug(f"Total words: {summary_len}")

        if (n_swaps is None and summary_perc is None) or (n_swaps is not None and summary_perc is not None):
            raise ValueError('You must specify exactly one of n_swaps or summary_perc.')

        if n_swaps is not None:
            n_words2swap = n_swaps*2  # Number of words to be swapped
            if not isinstance(n_swaps, int):
                raise TypeError('n_swaps must be an integer.')
            if n_swaps < 1 or n_words2swap > summary_len:
                raise ValueError('n_swaps must be a positive integer and the number of words'
                                 'resulting from the n_swaps (n_swaps*2) must be less than the summary length.')
            target_number = n_words2swap

        elif summary_perc is not None:
            if not isinstance(summary_perc, float):
                raise TypeError('summary_perc must be a float.')
            if not (0 < summary_perc < 1):
                raise ValueError('summary_perc must be positive and less than 1.')
            target_number = (int(summary_len * summary_perc) // 2) * 2
        logger.debug(f"Number of words to be swapped: {target_number}")

        # Get the indices of all the words
        summary_indices = [index for index in range(summary_len-1)]
        return summary_indices, target_number

    def random_swap_words(
            self,
            n_swaps: Optional[int] = None,
            summary_perc: Optional[float] = None
            ) -> str:
        """Swaps words from the summary n_swap times.

        Args:
            n_swaps (Optional[int]): The number of swaps to occur.
            summary_perc (Optional[float]): The percentage of the summary to be modified.

        Returns:
            str: The summary with the swapped words.

        """
        split_summary = self.input_summary.split()

        # Get summary indices and target number for swapping
        summary_indices, target_number = self._get_swap_indices(n_swaps, summary_perc)

        # Sample n_words2swap number of words indices
        word_indices = sample(summary_indices, k=target_number)
        logger.debug(f"Indices to be swapped: {word_indices}")

        # Swap the words and slide indices from left to the right
        word_indices_len = len(word_indices)
        index1, index2 = 0, 1

        while index2 <= word_indices_len:
            split_summary[word_indices[index1]], split_summary[word_indices[index2]] = split_summary[word_indices[index2]], split_summary[word_indices[index1]]
            logger.debug(f"Indices: {word_indices[index1]}, {word_indices[index2]} = {word_indices[index2]}, {word_indices[index1]}")
            logger.debug(f"Words: {split_summary[word_indices[index1]]}, {split_summary[word_indices[index2]]} = {split_summary[word_indices[index2]]}, {split_summary[word_indices[index1]]}")
            index1 += 2
            index2 += 2
        return ' '.join(split_summary)

    def consecutive_swap_words(
            self,
            n_swaps: Optional[int] = None,
            summary_perc: Optional[float] = None
    ) -> str:
        """Consecutively swaps words from the summary n_swap times.

        Args:
            n_swaps (Optional[int]): The number of swaps to occur.
            summary_perc (Optional[float]): The percentage of the summary to be modified.

        Returns:
            str: The summary with the swapped words.

        """
        split_summary = self.input_summary.split()

        # Get summary indices and target number for swapping
        summary_indices, target_number = self._get_swap_indices(n_swaps, summary_perc)

        # Sample n_swaps number of words indices
        word_indices = sample(summary_indices[:-1], k=target_number)
        logger.debug(f"Indices to be swapped: {word_indices}")

        # Swap the words and slide indices from left to the right
        for index in word_indices:
            if index < len(word_indices)-1:
                split_summary[word_indices[index]], split_summary[word_indices[index + 1]] = split_summary[word_indices[index + 1]], split_summary[word_indices[index]]
                logger.debug(f"Indices: {word_indices[index]}, {word_indices[index + 1]} = {word_indices[index + 1]}, {word_indices[index]}")
                logger.debug(f"Words: {split_summary[word_indices[index]]}, {split_summary[word_indices[index + 1]]} = {split_summary[word_indices[index + 1]]}, {split_summary[word_indices[index]]}")
        return ' '.join(split_summary)

    def remove_words(
            self,
            n_words: Optional[int] = None,
            summary_perc: Optional[float] = None
            ) -> str:
        """Removes n_words from the summary.

        Args:
            n_words (Optional[int]): Number of words to delete from the summary.
            summary_perc (Optional[float]): The percentage of the summary to be modified.

        Returns:
            str: The summary with the deleted words.

        """
        split_summary = self.input_summary.split()
        summary_len = len(split_summary)
        logger.debug(f"Total words: {summary_len}")

        if (n_words is None and summary_perc is None) or (n_words is not None and summary_perc is not None):
            raise ValueError('You must specify exactly one of n_words or summary_perc.')

        if n_words is not None:
            if not isinstance(n_words, int):
                raise TypeError('n_words must be an integer.')
            if not (0 < n_words < summary_len):
                raise ValueError('n_words must be a positive integer.')
            target_number = n_words

        if summary_perc is not None:
            if not isinstance(summary_perc, float):
                raise TypeError('summary_perc must be a float.')
            if not (0 < summary_perc < 1):
                raise ValueError('summary_perc must be positive and less than 1.')
            target_number = int(summary_len * summary_perc)
        logger.debug(f"Number of words to be removed: {target_number}")

        logger.debug("Words removed:")
        for _ in range(target_number):
            random_word = choice(split_summary)
            logger.debug(f"{_}: {random_word}")
            split_summary.remove(random_word)
        return ' '.join(split_summary)

    def remove_sentence(
            self,
            n_sentences: Optional[int] = None,
            summary_perc: Optional[float] = None
            ) -> str:
        """Removes a sentence from the summary.

        Args:
            n_sentences (Optional[int]): The number of sentences to be removed.
            summary_perc (Optional[float]): The percentage of the summary to be modified.

        Returns:
            str: The summary after the removal of the sentence.

        """
        doc = nlp(self.input_summary)
        sentences = [sent.text.strip() for sent in doc.sents]
        summary_len = len(sentences)
        logger.debug(f"Total sentences: {summary_len}")

        if (n_sentences is None and summary_perc is None) or (n_sentences is not None and summary_perc is not None):
            raise ValueError('You must specify exactly one of n_words or summary_perc.')

        if n_sentences is not None:
            if not isinstance(n_sentences, int):
                raise TypeError('n_sentences must be an integer.')
            if not (0 < n_sentences < summary_len):
                raise ValueError('n_sentences must be a positive integer and less in size than the length of the summary.')
            target_number = n_sentences

        if summary_perc is not None:
            if not isinstance(summary_perc, float):
                raise TypeError('summary_perc must be a float.')
            if not (0 < summary_perc < 1):
                raise ValueError('summary_perc must be positive and less than 1.')
            target_number = int(summary_len * summary_perc)
        logger.debug(f"Number of sentences to be removed: {target_number}")

        # Remove sentences
        sents_remove = choices(sentences, k=target_number)
        logger.debug("Sentences to be removed:")
        for i, sent in enumerate(sents_remove):
            logger.debug(f"{i}:\n{sent}\n")
        new_text = [sent for sent in sentences if sent not in sents_remove]
        return ''.join(new_text)

    def insert_sentence(
            self,
            target: str,
            source_docs: list,
            gold_dir: str,
            n_sentences: Optional[int] = None,
            summary_perc: Optional[float] = None
            ) -> str:
        """Inserts a sentence from another summary in the dataset.

        Args:
            target (str): The number of the summary.
            source_docs (list): The list of annual reports in the analysis.
            gold_dir (str): The path of the gold summaries.
            n_sentences (Optional[int]): The number of sentences to be inserted.
            summary_perc (Optional[float]): The percentage of the summary to be modified.

        Returns:
            str: The summary with the new inserted sentence.

        """
        doc = nlp(self.input_summary)
        sentences = [sent.text.strip() for sent in doc.sents]
        summary_len = len(sentences)
        logger.debug(f"Total sentences: {summary_len}")

        # Choose another summary
        remaining_summaries = [doc for doc in source_docs if doc != target]
        rand_summary = choice(remaining_summaries)
        logger.debug(f"Random summary: {rand_summary}")

        # Choose a sentence from the randomly picked summary to insert to the original one
        with open(os.path.join(gold_dir, f'{rand_summary}_1.txt'), mode='r', encoding='utf-8') as file:
            rand_gold_summary = file.read()
            rand_doc = nlp(rand_gold_summary)
            rand_sentences = [sent.text.strip() for sent in rand_doc.sents]
            rand_sentence_len = len(rand_sentences)
            logger.debug(f"Random summary sentences: {rand_sentence_len}")

            if (n_sentences is None and summary_perc is None) or (n_sentences is not None and summary_perc is not None):
                raise ValueError('You must specify exactly one of n_sentences or summary_perc.')

            if n_sentences is not None:
                if not isinstance(n_sentences, int):
                    raise TypeError('n_sentences must be an integer.')
                if not (0 < n_sentences < rand_sentence_len):
                    raise ValueError('n_sentences must be a positive integer and less in size than the length of the random summary.')
                target_number = n_sentences

            if summary_perc is not None:
                if not isinstance(summary_perc, float):
                    raise TypeError('summary_perc must be a float.')
                if not (0 < summary_perc < 1):
                    raise ValueError('summary_perc must be positive and less than 1.')
                target_number = int(summary_len * summary_perc)
            logger.debug(f"Number of sentences to insert: {target_number}")

            # Insert new sentences
            sents_insert = choices(sentences, k=target_number)
            logger.debug("Sentences to be inserted:")
            for i, sent in enumerate(sents_insert):
                logger.debug(f"{i}:\n{sent}\n")
            original_text = ''.join([sent for sent in sentences])
            new_text = ''.join(sents_insert)
            return new_text + original_text

    def repeat_sentence(
            self,
            n_repeats: Optional[int] = None,
            summary_perc: Optional[float] = None
            ) -> str:
        """Repeats a sentence in the summary.

        Args:
            n_repeats (int): The number of times the sentence should be repeated.
            summary_perc (Optional[float]): The percentage of the summary to be modified.

        Returns:
            str: The summary with the repeated sentences.

        """
        doc = nlp(self.input_summary)
        sentences = [sent.text.strip() for sent in doc.sents]
        summary_len = len(sentences)
        logger.debug(f"Total sentences: {summary_len}")

        if (n_repeats is None and summary_perc is None) or (n_repeats is not None and summary_perc is not None):
            raise ValueError('You must specify exactly one of n_repeats or summary_perc.')

        if n_repeats is not None:
            if not isinstance(n_repeats, int):
                raise TypeError('n_repeats must be an integer.')
            if n_repeats < 1:
                raise ValueError('n_repeats must be a positive integer.')
            target_number = n_repeats

        if summary_perc is not None:
            if not isinstance(summary_perc, float):
                raise TypeError('summary_perc must be a float.')
            if not (0 < summary_perc < 1):
                raise ValueError('summary_perc must be positive and less than 1.')
            target_number = int(summary_len * summary_perc)
        logger.debug(f"Number of times to repeat: {target_number}")

        sent_repeat = choice(sentences)
        logger.debug(f"Sentence to be repeated: {sent_repeat}")
        original_text = ''.join([sent for sent in sentences])
        new_text = ''.join([sent_repeat] * target_number)
        return new_text + original_text
