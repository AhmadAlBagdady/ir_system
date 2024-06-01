import jsonlines
import pandas as pd

import pickle

from text_processors.remove_punctuations_processor import RemovePunctuationsProcessor
from text_processors.remove_stopwords_processor import RemoveStopwordsProcessor
from text_processors.spell_checker_processor import SpellCheckerProcessor
from text_processors.stemmer_processor import StemmerProcessor
from text_processors.word_tokenize_processor import WordTokenizeProcessor
from text_processors.lemmatizer_processor import LemmatizerProcessor
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict


def load_csv(file_path):
    df = pd.read_csv(file_path, usecols=['id_right', 'text_right'], nrows=100)
    df['processed'] = df['text_right'].apply(csv_processing)
    return df


def csv_processing(text):
    text = RemovePunctuationsProcessor(text).get_tokens()
    text = WordTokenizeProcessor(text).get_tokens()
    # text = SpellCheckerProcessor(text).get_tokens()
    text = RemoveStopwordsProcessor(text).get_tokens()
    text = LemmatizerProcessor(text).get_tokens()
    text = StemmerProcessor(text).get_tokens()
    return ' '.join(text)


def create_tfidf_features(data):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['processed'])
    return vectorizer, X, data


csv_data = load_csv('/home/baraa/Desktop/wikIR1k/documents.csv')

# Example usage, assuming 'processed_texts' is a list of preprocessed documents
tfidf_vectorizer, tfidf_matrix, full_data = create_tfidf_features(csv_data)


def build_inverted_index(tfidf_matrix, tfidf_vectorizer):
    inverted_index = defaultdict(list)
    for doc_index, doc in enumerate(tfidf_matrix):
        for word_index, weight in zip(doc.indices, doc.data):
            inverted_index[tfidf_vectorizer.get_feature_names_out()[word_index]].append((doc_index, weight))
    return inverted_index


index = build_inverted_index(tfidf_matrix, tfidf_vectorizer)


def save_index(index, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(index, f)


save_index(index, 'inverted_index_csv.pkl')


def load_index(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


vectorizer = load_index('inverted_index_csv.pkl')


def preprocess_query(query):
    query = RemovePunctuationsProcessor(query).get_tokens()
    query = WordTokenizeProcessor(query).get_tokens()
    query = SpellCheckerProcessor(query).get_tokens()
    query = RemoveStopwordsProcessor(query).get_tokens()
    query = LemmatizerProcessor(query).get_tokens()
    query = StemmerProcessor(query).get_tokens()
    return ' '.join(query)


def vectorize_query(query):
    preprocessed_query = preprocess_query(query)
    # print(query)
    # quit()
    query_vector = tfidf_vectorizer.transform([preprocessed_query])  # Note: transform expects an iterable
    return query_vector


def search_documents(query_vector, tfidf_matrix):
    from sklearn.metrics.pairwise import cosine_similarity
    # Calculate cosine similarity between the query vector and all document vectors
    similarities = cosine_similarity(query_vector, tfidf_matrix)
    # Sort document indices by similarity scores in descending order
    top_indices = similarities.argsort()[0][::-1]
    return top_indices


def print_top_documents(top_indices, data, num_results=10):
    top_docs = data.iloc[top_indices[:num_results]]
    for index, row in top_docs.iterrows():
        print(f"Document ID: {row['id_right']}, Content: {row['text_right']}\n")


# Example query
user_query = "columbia"

# Vectorize the query
query_vector = vectorize_query(user_query)

# Search documents
top_document_indices = search_documents(query_vector, tfidf_matrix)
print("Top document indices:", print_top_documents(top_document_indices,full_data)[:10])  # Show top 10 results
