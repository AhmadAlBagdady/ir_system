import jsonlines
import pandas as pd

import pickle

from text_processors.remove_punctuation_processor import RemovePunctuationProcessor
from text_processors.remove_stopwords_processor import RemoveStopwordsProcessor
from text_processors.spell_checker_processor import SpellCheckerProcessor
from text_processors.stemmer_processor import StemmerProcessor
from text_processors.word_tokenize_processor import WordTokenizeProcessor
from text_processors.lemmatizer_processor import LemmatizerProcessor
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict


def load_csv(file_path):
    df = pd.read_csv(file_path, usecols=['id_right', 'text_right'], nrows=1000)
    data = df['text_right'].apply(csv_processing)
    return data


def csv_processing(text):
    text = RemovePunctuationProcessor(text).get_tokens()
    text = WordTokenizeProcessor(text).get_tokens()
    # text = SpellCheckerProcessor(text).get_tokens()
    text = RemoveStopwordsProcessor(text).get_tokens()
    text = LemmatizerProcessor(text).get_tokens()
    text = StemmerProcessor(text).get_tokens()
    return ' '.join(text)


def create_tfidf_features(corpus):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    return vectorizer, X


# Example usage, assuming 'processed_texts' is a list of preprocessed documents
corpus = ["your first document text here", "your second document document  document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document document text here", ]
tfidf_vectorizer, tfidf_matrix = create_tfidf_features(corpus)


def build_inverted_index(tfidf_matrix, tfidf_vectorizer):
    inverted_index = defaultdict(list)
    for doc_index, doc in enumerate(tfidf_matrix):
        for word_index, weight in zip(doc.indices, doc.data):
            inverted_index[tfidf_vectorizer.get_feature_names_out()[word_index]].append((doc_index, weight))
    return inverted_index


print(build_inverted_index(tfidf_matrix, tfidf_vectorizer))


# csv_data = load_csv('/home/baraa/Desktop/wikIR1k/documents.csv')

# print(csv_data)


# print(load_jsonl('/home/baraa/Desktop/TREC-TOT/corpus.jsonl'))


# print(RemovePunctuationProcessor(s).get_tokens())

def dd(text):
    print(text)
    quit()
