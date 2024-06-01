import pandas as pd
import jsonlines
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from six import print_
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import pickle

from text_processors.remove_punctuation_processor import RemovePunctuationProcessor
from text_processors.remove_stopwords_processor import RemoveStopwordsProcessor
from text_processors.spell_checker_processor import SpellCheckerProcessor
from text_processors.stemmer_processor import StemmerProcessor
from text_processors.word_tokenize_processor import WordTokenizeProcessor


def load_csv(file_path):
    df = pd.read_csv(file_path, usecols=['text_right'], nrows=10)
    ahmad = csv_builder_processor('Ahmed loves banana, will you try anything')
    return ahmad


def csv_builder_processor(text):
    removePunctuation = RemovePunctuationProcessor(text)
    wordTokenizeProcessor = WordTokenizeProcessor(text)
    removeStopWordsProcessor = RemoveStopwordsProcessor(wordTokenizeProcessor.tokens)
    stemmerProcessor = StemmerProcessor(removeStopWordsProcessor.tokens)
    spellCheckerProcessor = SpellCheckerProcessor(stemmerProcessor.tokens)

    return spellCheckerProcessor.get_tokens()


# def csv_text_processing(text):


print(load_csv('/home/baraa/Desktop/wikIR1k/documents.csv'))