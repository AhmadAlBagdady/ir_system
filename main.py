from text_processors.word_tokenize_processor import TokenizationProcessor
from text_processors.remove_stopwords_processor import RemoveStopwordsProcessor
from text_processors.stemmer_processor import StemmingProcessor
from text_processors.lemmatizer_processor import LemmatizationProcessor
from text_processors.spell_checker_processor import SpellCheckerProcessor
from text_processors.remove_punctuation_processor import RemovePunctuationProcessor

# tokenization = TokenizationProcessor('This is a sampli sentinse withh speling erors. The boys are running and the leaves are falling. could you find a way to let me down slowely. The boys are running and the leaves are falling a little sempethy I hope you can show me')
# stopWords = RemoveStopwordsProcessor(tokenization.tokens)
# stemmed = StemmingProcessor(stopWords.tokens)
# lemmatizer = LemmatizationProcessor(stopWords.tokens)
# spellChecker = SpellCheckerProcessor(stopWords.tokens)
removePunc = RemovePunctuationProcessor("'Hello, world! This is some sample' text.''")
print(removePunc.get_tokens())