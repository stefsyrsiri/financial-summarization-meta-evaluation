import os
from loguru import logger
from random import choice, sample, shuffle


class SummaryDestructor:
    def __init__(self, input_summary: str):
        self.input_summary = input_summary
        logger.info("SummaryDestructor initialized.")

    def shuffle_words(self) -> str:
        split_summary = self.input_summary.split()
        shuffle(split_summary)
        return ' '.join(split_summary)

    def swap_words(self, n_swaps: int = 1) -> str:
        split_summary = self.input_summary.split()
        summary_len = len(split_summary)
        n_words2swap = n_swaps*2
        if n_words2swap <= summary_len:
            summary_indices = [index for index in range(summary_len-1)]
            word_indices = sample(summary_indices, k=n_words2swap)
            word_indices.sort()
            word_indices_len = len(word_indices)
            index1, index2 = 0, 1
            while index2 <= word_indices_len:
                split_summary[word_indices[index1]], split_summary[word_indices[index2]] = split_summary[word_indices[index2]], split_summary[word_indices[index1]]
                index1 += 2
                index2 += 2
        return ' '.join(split_summary)

    def remove_words(self, n_words: int) -> str:
        split_summary = self.input_summary.split()
        if n_words < len(split_summary):
            for _ in range(n_words):
                random_word = choice(split_summary)
                split_summary.remove(random_word)
            return ' '.join(split_summary)

    def remove_sentence(self) -> str:
        sentences = self.input_summary.split('.')
        sent_remove_len = 0
        while sent_remove_len < 2:
            sent_remove = choice(sentences)
            sent_remove_len = len(sent_remove)
        new_text = [sent for sent in sentences if sent != sent_remove]
        return ''.join(new_text)

    def insert_sentence(self, target, source_docs, gold_dir) -> str:
        sentences = self.input_summary.split('.')

        # Choose another summary
        remaining_summaries = [doc for doc in source_docs if doc != target]
        rand_summary = choice(remaining_summaries)

        # Choose a sentence from the randomly picked summary to insert to the original one
        with open(os.path.join(gold_dir, f'{rand_summary}_1.txt'), mode='r', encoding='utf-8') as file:
            rand_gold_summary = file.read()
            rand_sentences = rand_gold_summary.split('.')
            sent_insert_len = 0
            while sent_insert_len < 2:
                sent_insert = choice(rand_sentences)
                sent_insert_len = len(sent_insert)

        new_text = [sent for sent in sentences]

        return ''.join(new_text) + '' + sent_insert

    def repeat_sentence(self, n_repeats) -> str:
        sentences = self.input_summary.split('.')
        sent_repeat_len = 0
        while sent_repeat_len < 2:
            sent_repeat = choice(sentences)
            sent_repeat_len = len(sent_repeat)

        repeated_sent = [sent_repeat] * n_repeats
        new_text = [sent for sent in sentences]

        return ''.join(new_text) + '' + ''.join(repeated_sent)
