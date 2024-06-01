from text_processors.base_processor import BaseProcessor
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet

class LemmatizerProcessor(BaseProcessor):
    def __init__(self,inputs:list):
        self.inputs=inputs
        
        self.__lemmatization()
        pass

    def __get_wordnet_pos(self,tag_parameter):
        tag = tag_parameter[0].upper()
        tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    
        return tag_dict.get(tag, wordnet.NOUN)
    
    def __lemmatization(self):
        pos_tags = pos_tag(self.inputs)
        lemmatizer = WordNetLemmatizer()
        self.tokens = [lemmatizer.lemmatize(word, pos=self.__get_wordnet_pos(tag)) for word, tag in pos_tags]