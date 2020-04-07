from pii_recognition.tokenisation.tokenisers import nltk_word_tokenizer

from .registry import Registry


class TokeniserRegistry(Registry):
    def add_predefines(self):
        self.add_item(nltk_word_tokenizer)
