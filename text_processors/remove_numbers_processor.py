from text_processors.base_processor import BaseProcessor
import re


class RemoveNumbersProcessor(BaseProcessor):
    def __init__(self, inputs: str):
        self.inputs = inputs

        self.__remove_numbers()
        pass

    def __remove_numbers(self):
        number_pattern = r'[0-9]+'
        self.tokens = re.sub(number_pattern, '', self.inputs)
        
