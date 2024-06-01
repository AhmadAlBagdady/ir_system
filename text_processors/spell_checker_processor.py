from text_processors.base_processor import BaseProcessor
from spellchecker import SpellChecker


class SpellCheckerProcessor(BaseProcessor):
    def __init__(self, inputs: list):
        self.inputs = inputs

        self.__correct_sentence_spelling()
        pass

    def __correct_sentence_spelling(self):
        spell = SpellChecker()
        misspelled = spell.unknown(self.inputs)
        for i, token in enumerate(self.inputs):
            if token in misspelled:
                corrected = spell.correction(token)
                if corrected is not None:
                    self.inputs[i] = corrected
        self.tokens = self.inputs
