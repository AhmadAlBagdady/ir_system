from nltk.corpus import stopwords
from text_processors.base_processor import BaseProcessor


class RemoveStopwordsProcessor(BaseProcessor):
     def __init__(self,inputs:list):
        self.inputs=inputs
        self.stopwords = stopwords.words('english')

        self.__remove_stopwords()
        pass
     
     def __remove_stopwords(self):
         for token in self.inputs:
             if token in self.stopwords:
                 self.inputs.remove(token)        

         self.tokens=self.inputs
