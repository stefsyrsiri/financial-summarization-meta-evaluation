"""
This module provides tools that destroy summaries by:

- shuffling their words
- removing words
- removing sentences
- inserting sentences from other summaries (from the same dataset)
- repeating sentences
"""


import os
from loguru import logger
from random import choice, sample, shuffle
from typing import Optional


class SummaryDestructor:
    def __init__(self, input_summary: str):
        """Makes summaries noisy.

        Args:
            input_summary (str): The string of the summary to be destroyed.

        """
        self.input_summary = input_summary
        logger.info("SummaryDestructor initialized.")

    def shuffle_words(self) -> str:
        """Shuffles the summary's words.

        Returns:
            str: The shuffled summary.

        """
        split_summary = self.input_summary.split()
        shuffle(split_summary)
        return ' '.join(split_summary)

    def swap_words(
            self,
            n_swaps: Optional[int],
            summary_perc: Optional[float]
            ) -> str:
        """Swaps words from the summary n_swap times.

        Args:
            n_swaps (Optional[int]): The number of swaps to occur.
            summary_perc (Optional[float]): The percentage of the summary to be modified.

        Returns:
            str: The summary with the swapped words.

        """
        split_summary = self.input_summary.split()
        summary_len = len(split_summary)

        if (n_swaps is None and summary_perc is None) or (n_swaps is not None and summary_perc is not None):
            raise ValueError('You must specify exactly one of n_swaps or summary_perc.')

        if n_swaps is not None:
            n_words2swap = n_swaps*2  # Number of words to be swapped
            if not isinstance(n_swaps, int):
                raise TypeError('n_swaps must be an integer.')
            if n_swaps < 1 or n_words2swap > summary_len:
                raise ValueError('n_swaps must be a positive integer and the number of words'
                                 'resulting from the n_swaps (n_swaps*2) must be less than the summary length.')
        else:
            target_number = n_words2swap

        if summary_perc is not None:
            if not isinstance(summary_perc, float):
                raise TypeError('summary_perc must be a float.')
            if not (0 < summary_perc < 1):
                raise ValueError('summary_perc must be positive and less than 1.')
        else:
            target_number = (int(summary_len * summary_perc) // 2) * 2

        # Get the indices of all the words
        summary_indices = [index for index in range(summary_len-1)]

        # Sample n_words2swap number of words indices
        word_indices = sample(summary_indices, k=target_number)

        # Sort them so that swaps will be in sequential pairs of two
        word_indices.sort()
        word_indices_len = len(word_indices)
        index1, index2 = 0, 1

        # Swap the words and slide indices from left to the right
        while index2 <= word_indices_len:
            split_summary[word_indices[index1]], split_summary[word_indices[index2]] = split_summary[word_indices[index2]], split_summary[word_indices[index1]]
            index1 += 2
            index2 += 2
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

        if (n_words is None and summary_perc is None) or (n_words is not None and summary_perc is not None):
            raise ValueError('You must specify exactly one of n_words or summary_perc.')

        if n_words is not None:
            if not isinstance(n_words, int):
                raise TypeError('n_words must be an integer.')
            if not (0 < n_words < summary_len):
                raise ValueError('n_words must be a positive integer.')
        else:
            target_number = n_words

        if summary_perc is not None:
            if not isinstance(summary_perc, float):
                raise TypeError('summary_perc must be a float.')
            if not (0 < summary_perc < 1):
                raise ValueError('summary_perc must be positive and less than 1.')
        else:
            target_number = int(summary_len * summary_perc)

        for _ in range(target_number):
            random_words = choice(split_summary, k=target_number)
            split_summary.remove(random_words)
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
        sentences = self.input_summary.split('.')
        summary_len = len(sentences)

        if (n_sentences is None and summary_perc is None) or (n_sentences is not None and summary_perc is not None):
            raise ValueError('You must specify exactly one of n_words or summary_perc.')

        if n_sentences is not None:
            if not isinstance(n_sentences, int):
                raise TypeError('n_sentences must be an integer.')
            if not (0 < n_sentences < summary_len):
                raise ValueError('n_sentences must be a positive integer and less in size than the length of the summary.')
        else:
            target_number = n_sentences

        if summary_perc is not None:
            if not isinstance(summary_perc, float):
                raise TypeError('summary_perc must be a float.')
            if not (0 < summary_perc < 1):
                raise ValueError('summary_perc must be positive and less than 1.')
        else:
            target_number = int(summary_len * summary_perc)

        # Remove sentences
        sents_remove = choice(sentences, k=target_number)
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
        sentences = self.input_summary.split('.')
        summary_len = len(sentences)

        # Choose another summary
        remaining_summaries = [doc for doc in source_docs if doc != target]
        rand_summary = choice(remaining_summaries)

        # Choose a sentence from the randomly picked summary to insert to the original one
        with open(os.path.join(gold_dir, f'{rand_summary}_1.txt'), mode='r', encoding='utf-8') as file:
            rand_gold_summary = file.read()
            rand_sentences = rand_gold_summary.split('.')
            rand_sentence_len = len(rand_sentences)

            if (n_sentences is None and summary_perc is None) or (n_sentences is not None and summary_perc is not None):
                raise ValueError('You must specify exactly one of n_sentences or summary_perc.')

            if n_sentences is not None:
                if not isinstance(n_sentences, int):
                    raise TypeError('n_sentences must be an integer.')
                if not (0 < n_sentences < rand_sentence_len):
                    raise ValueError('n_sentences must be a positive integer and less in size than the length of the random summary.')
            else:
                target_number = n_sentences

            if summary_perc is not None:
                if not isinstance(summary_perc, float):
                    raise TypeError('summary_perc must be a float.')
                if not (0 < summary_perc < 1):
                    raise ValueError('summary_perc must be positive and less than 1.')
            else:
                target_number = int(summary_len * summary_perc)

            # Insert new sentences
            sents_insert = choice(sentences, k=target_number)
            original_text = ''.join([sent for sent in sentences])
            new_text = ''.join(sents_insert)
            return original_text + new_text

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
        sentences = self.input_summary.split('.')
        summary_len = len(sentences)

        if (n_repeats is None and summary_perc is None) or (n_repeats is not None and summary_perc is not None):
            raise ValueError('You must specify exactly one of n_repeats or summary_perc.')

        if n_repeats is not None:
            if not isinstance(n_repeats, int):
                raise TypeError('n_repeats must be an integer.')
            if n_repeats < 1:
                raise ValueError('n_repeats must be a positive integer.')
        else:
            target_number = n_repeats

        if summary_perc is not None:
            if not isinstance(summary_perc, float):
                raise TypeError('summary_perc must be a float.')
            if not (0 < summary_perc < 1):
                raise ValueError('summary_perc must be positive and less than 1.')
        else:
            target_number = int(summary_len * summary_perc)

        sents_repeat = choice(sentences, k=target_number)
        original_text = ''.join([sent for sent in sentences])
        new_text = ''.join([sents_repeat] * n_repeats)
        return original_text + new_text
