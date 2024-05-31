from nltk.tokenize import word_tokenize

class TokenizationProcessing:
    def __init__(self,text):
        self.text=text
        self.__tokenization()
        pass

    def __tokenization(self):
        self.tokens = word_tokenize(self.text)

    def get_tokens(self):
        return self.tokens