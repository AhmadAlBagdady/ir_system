from nltk.tokenize import sent_tokenize
from text_processors.base_processor import BaseProcessor


class SentTokenizeProcessor(BaseProcessor):
    def __init__(self, inputs: str):
        self.inputs = inputs
        self.__sent_tokenize()
        pass

    def __sent_tokenize(self):
        self.tokens = sent_tokenize(self.inputs)
