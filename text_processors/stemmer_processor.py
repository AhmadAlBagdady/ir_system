from nltk.stem import PorterStemmer
from text_processors.base_processor import BaseProcessor

class StemmerProcessor(BaseProcessor):
     def __init__(self,inputs:list):
        self.inputs=inputs
        self.stemmed_words = []

        self.stemmer = PorterStemmer()

        self.__stemming()
        pass
     
     def __stemming(self):
         self.tokens = [self.stemmer.stem(token) for token in self.inputs]
