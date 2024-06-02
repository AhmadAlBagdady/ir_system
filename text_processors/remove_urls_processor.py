from text_processors.base_processor import BaseProcessor
import re


class RemoveUrlsProcessor(BaseProcessor):
    def __init__(self, inputs: str):
        self.inputs = inputs

        self.__remove_urls()
        pass

    def __remove_urls(self):
        url_pattern = r'https?://\S+|www\.\S+'
        self.tokens = re.sub(url_pattern, '', self.inputs)
        
