from nltk.corpus import stopwords
from text_processing.tokenization_processing import TokenizationProcessing

class StopwordsProcessing:
     def __init__(self,tokenizationProcessing:TokenizationProcessing):
        self.tokenizationProcessing=tokenizationProcessing
        self.stopwords = stopwords.words('english')

        self.__remove_stopwords()
        pass
     
     def __remove_stopwords(self):
         for token in self.tokenizationProcessing.tokens:
             if token in self.stopwords:
                 self.tokenizationProcessing.tokens.remove(token)

     def get_tokens(self):
         return self.tokenizationProcessing.get_tokens()