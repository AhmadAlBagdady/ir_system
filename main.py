from text_processing.tokenization_processing import TokenizationProcessing
from text_processing.stopwords_processing import StopwordsProcessing

tokenization = TokenizationProcessing('could you find a way to let me down slowely.')
stopWords = StopwordsProcessing(tokenization)


print(stopWords.get_tokens())