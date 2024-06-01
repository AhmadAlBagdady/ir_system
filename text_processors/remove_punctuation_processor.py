from text_processors.base_processor import BaseProcessor
import string


class RemovePunctuationProcessor(BaseProcessor):
    def __init__(self, inputs: str):
        self.inputs = inputs

        self.__remove_punctuation()
        pass

    def __remove_punctuation(self):
        self.tokens = self.inputs.translate(str.maketrans('', '', string.punctuation))
