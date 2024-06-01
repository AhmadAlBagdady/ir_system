from nltk.tokenize import word_tokenize
from text_processors.base_processor import BaseProcessor


class WordTokenizeProcessor(BaseProcessor):
    def __init__(self, inputs: str):
        self.inputs = inputs
        self.__word_tokenize()
        pass

    def __word_tokenize(self):
        self.tokens = word_tokenize(self.inputs)
