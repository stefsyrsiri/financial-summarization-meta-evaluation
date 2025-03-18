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
from random import choice, sample  # sampling without replacement

nlp = spacy.load('el_core_news_sm')
nlp.add_pipe('sentencizer', before='parser')


class SummaryDestructor:
    def __init__(
            self,
            input_summary: str,
            noise_percentage: float
            ):
        """Makes summaries noisy.

        Args:
            input_summary (str): The string of the summary to be destroyed.
            noise_percentage (float): The percentage of the summary to be modified.

        """
        if not (0 < noise_percentage < 1):
            raise ValueError('summary_perc must be positive and less than 1.')
        self.input_summary = input_summary
        self._noise_percentage = noise_percentage
        self._word_sample_space = [index for index in range(len(self.input_summary.split()) - 1)]
        self._sentence_sample_space = [sent.text.strip() for sent in nlp(self.input_summary).sents]
        logger.info(f"SummaryDestructor initialized with {self.noise_percentage:.0%} noise.")

    @property
    def noise_percentage(self):
        return self._noise_percentage

    @noise_percentage.setter
    def noise_percentage(self, new_percentage: float):
        if isinstance(new_percentage, float) and 0 < new_percentage < 1:
            self._noise_percentage = new_percentage

    @property
    def word_sample_space(self):
        return self._word_sample_space

    @word_sample_space.setter
    def word_sample_space(self, reduced_sample_space: list[int]):
        if isinstance(reduced_sample_space, list[int]) and len(reduced_sample_space) < self.word_sample_space:
            self._word_sample_space = reduced_sample_space

    @property
    def sentence_sample_space(self):
        return self._sentence_sample_space

    @sentence_sample_space.setter
    def sentence_sample_space(self, reduced_sample_space: list[str]):
        if isinstance(reduced_sample_space, list[str]) and len(reduced_sample_space) < self.sentence_sample_space:
            self._sentence_sample_space = reduced_sample_space

    # Helper for the random_swap and consecutive_swap methods
    def _get_swap_indices(self):
        """Helper method for the random_swap_words and consecutive_swap_words methods.
           Prepares the summary indices and target number used for swapping.

        Returns:
            The summary indices and the target number used for sampling summary indices.

        """
        split_summary = self.input_summary.split()
        summary_len = len(split_summary)
        logger.debug(f"Total words: {summary_len}")

        target_number = int(summary_len * self.noise_percentage) // 2
        logger.debug(f"Number of swaps: {target_number}")
        return target_number

    def random_swap_words(self) -> str:
        """Swaps words from the summary n_swap times.

        Returns:
            str: The summary with the swapped words.

        """
        split_summary = self.input_summary.split()

        # Get summary indices and target number for swapping
        target_number = self._get_swap_indices()

        # Sample n_words2swap number of words indices
        word_indices = sample(self.word_sample_space, k=target_number * 2)
        logger.debug(f"Number of indices to be swapped: {target_number * 2}")
        logger.debug(f"Indices to be swapped: {word_indices}")

        # Swap the words and slide indices from left to the right
        word_indices_len = len(word_indices)
        index1, index2 = 0, 1

        while index2 <= word_indices_len:
            logger.debug(f"Indices: {word_indices[index1]}, {word_indices[index2]} = {word_indices[index2]}, {word_indices[index1]}")
            logger.debug(f"Words: {split_summary[word_indices[index1]]}, {split_summary[word_indices[index2]]} = {split_summary[word_indices[index2]]}, {split_summary[word_indices[index1]]}")
            split_summary[word_indices[index1]], split_summary[word_indices[index2]] = split_summary[word_indices[index2]], split_summary[word_indices[index1]]
            index1 += 2
            index2 += 2
        return ' '.join(split_summary)

    def consecutive_swap_words(self) -> str:
        """Consecutively swaps words from the summary n_swap times.

        Returns:
            str: The summary with the swapped words.

        """
        split_summary = self.input_summary.split()

        # Get summary indices and target number for swapping
        target_number = self._get_swap_indices()

        # Sample n_swaps number of words indices
        word_indices = sample(self.word_sample_space[:-1], k=target_number)
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
        return ' '.join(split_summary)

    def remove_words(self) -> str:
        """Removes n_words from the summary.

        Returns:
            str: The summary with the deleted words.

        """
        split_summary = self.input_summary.split()
        summary_len = len(split_summary)
        logger.debug(f"Total words: {summary_len}")

        target_number = int(summary_len * self.noise_percentage)
        logger.debug(f"Number of words to be removed: {target_number}")

        logger.debug("Words removed:")
        random_words = sample(split_summary, k=target_number)
        for i, random_word in enumerate(random_words):
            logger.debug(f"{i}: {random_word}")
            split_summary.remove(random_word)
        return ' '.join(split_summary)

    def remove_sentence(self) -> str:
        """Removes a sentence from the summary.

        Returns:
            str: The summary after the removal of the sentence.

        """
        summary_len = len(self.sentence_sample_space)
        logger.debug(f"Total sentences: {summary_len}")

        target_number = int(summary_len * self.noise_percentage)
        logger.debug(f"Number of sentences to be removed: {target_number}")

        # Remove sentences
        sents_remove = sample(self.sentence_sample_space, k=target_number)
        logger.debug("Sentences to be removed:")
        for i, sent in enumerate(sents_remove):
            logger.debug(f"{i}:\n{sent}\n")
        new_text = [sent for sent in self.sentence_sample_space if sent not in sents_remove]
        return ''.join(new_text)

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
        summary_len = len(self.sentence_sample_space)
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

            target_number = int(summary_len * self.noise_percentage)
            logger.debug(f"Number of sentences to insert: {target_number}")

            # Insert new sentences
            sents_insert = sample(self.sentence_sample_space, k=target_number)
            logger.debug("Sentences to be inserted:")
            for i, sent in enumerate(sents_insert):
                logger.debug(f"{i}:\n{sent}\n")
            original_text = ''.join([sent for sent in self.sentence_sample_space])
            new_text = ''.join(sents_insert)
            return new_text + original_text

    def repeat_sentence(self) -> str:
        """Repeats a sentence in the summary.

        Returns:
            str: The summary with the repeated sentences.

        """
        summary_len = len(self.sentence_sample_space)
        logger.debug(f"Total sentences: {summary_len}")

        target_number = int(summary_len * self.noise_percentage)
        logger.debug(f"Number of times to repeat: {target_number}")

        sent_repeat = choice(self.sentence_sample_space)
        logger.debug(f"Sentence to be repeated: {sent_repeat}")
        original_text = ''.join([sent for sent in self.sentence_sample_space])
        new_text = ''.join([sent_repeat] * target_number)
        return new_text + original_text
