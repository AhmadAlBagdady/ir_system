from text_processors.word_tokenize_processor import WordTokenizeProcessor
from text_processors.remove_stopwords_processor import RemoveStopwordsProcessor
from text_processors.stemmer_processor import StemmerProcessor
from text_processors.lemmatizer_processor import LemmatizerProcessor
from text_processors.spell_checker_processor import SpellCheckerProcessor
from text_processors.remove_punctuation_processor import RemovePunctuationProcessor

# tokenization = TokenizationProcessor('This is a sampli sentinse withh speling erors. The boys are running and the leaves are falling. could you find a way to let me down slowely. The boys are running and the leaves are falling a little sempethy I hope you can show me')
# stopWords = RemoveStopwordsProcessor(tokenization.tokens)
# stemmed = StemmingProcessor(stopWords.tokens)
# lemmatizer = LemmatizationProcessor(stopWords.tokens)
# spellChecker = SpellCheckerProcessor(stopWords.tokens)
removePunc = RemovePunctuationProcessor("'Hello, world! This is some sample' text.''")
print(removePunc.get_tokens())

# def load_csv(filepath):
#     data = pd.read_csv(filepath, usecols=['id_right', 'text_right'], nrows=40000)
#     data['processed'] = data['text_right'].apply(
#         preprocess_text)  # Replace 'text_column' with the actual text column name
#     return data

# csv_data = load_csv('C:\Users\Ahmad\Desktop\ir\dataset\wikIR1k\documents.csv')
# print('ahmad')
# print(csv_data)