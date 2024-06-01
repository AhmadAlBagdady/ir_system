from text_processors.base_processor import BaseProcessor
import string


class RemovePunctuationsProcessor(BaseProcessor):
    def __init__(self, inputs: str):
        self.inputs = inputs

        self.__remove_punctuations()
        pass

    def __remove_punctuations(self):
        self.tokens = self.inputs.translate(str.maketrans('', '', string.punctuation))
