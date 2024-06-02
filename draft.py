from builtins import list

import jsonlines
import pandas as pd

import pickle

from nltk import word_tokenize

from text_processors.remove_punctuations_processor import RemovePunctuationsProcessor
from text_processors.remove_stopwords_processor import RemoveStopwordsProcessor
from text_processors.spell_checker_processor import SpellCheckerProcessor
from text_processors.stemmer_processor import StemmerProcessor
from text_processors.word_tokenize_processor import WordTokenizeProcessor
from text_processors.lemmatizer_processor import LemmatizerProcessor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from collections import defaultdict
import numpy as np


def dd(text):
    print(text)
    quit()


def load_csv(file_path):
    df = pd.read_csv(file_path, usecols=['id_right', 'text_right'], nrows=1000)
    df.rename(columns={'id_right': 'doc_id', 'text_right': 'text'}, inplace=True)
    df.index = ['Document'] * len(df)

    return df


def csv_processing(text):
    text = RemovePunctuationsProcessor(text).get_tokens()
    text = WordTokenizeProcessor(text).get_tokens()
    # text = SpellCheckerProcessor(text).get_tokens()
    text = RemoveStopwordsProcessor(text).get_tokens()
    text = LemmatizerProcessor(text).get_tokens()
    text = StemmerProcessor(text).get_tokens()
    return ' '.join(text)



csv_data = load_csv('/home/baraa/Desktop/wikIR1k/documents.csv')


# print(csv_data)

def storing_indexing_terms():
    global indexing_terms_df
    indexing_terms = set()

    for doc in csv_data.values:
        indexing_terms.update(word_tokenize(doc[1]))
        indexing_terms_df = pd.DataFrame(indexing_terms)

    return indexing_terms_df

def create_tfidf_features(data):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['processed'])
    return vectorizer, X, data



# Example usage, assuming 'processed_texts' is a list of preprocessed


# index = storing_indexing_terms()
# print(index)


def build_inverted_index():
    inverted_index = defaultdict(list)

    for docId, doc in csv_data.values:
        doc_terms = set(word_tokenize(doc))
        for term in doc_terms:
            inverted_index[term].append(docId)

    return dict(inverted_index)


index = build_inverted_index()


print(index)

def calculate_tf_idf():
    documents = csv_data['text'].values
    # dd()
    # TODO : ADD YOUR OWN TOKENIZER & PREPROCESSOR !
    vectorizer = TfidfVectorizer()
    # Fit the vectorizer to the documents
    tfidf_matrix = vectorizer.fit_transform(documents)
    df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out(), index=csv_data['doc_id'].values)
    return df


tf_idf_df = calculate_tf_idf()
print(tf_idf_df)

# index = build_inverted_index()
# print(index)


# def save_index(index, file_name):
#     with open(file_name, 'wb') as f:
#         pickle.dump(index, f)
#
#
#
# def tf_idf(documents):
#     processed_documents = [csv_processing(doc) for doc in documents['text']]
#     vectorizer = TfidfVectorizer(max_features=10000, max_df=0.95, min_df=0.01, dtype=np.float32)
#     tfidf_matrix = vectorizer.fit_transform(processed_documents)
#     feature_names = vectorizer.get_feature_names_out()
#     inverted_index = build_inverted_index(tfidf_matrix, feature_names)
#     save_index(inverted_index, 'inverted_index_csv.pkl')
#     df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out(), index=documents.index)
#
#     return df, vectorizer
#
#
#
#
#
# def load_index(file_name):
#     with open(file_name, 'rb') as f:
#         return pickle.load(f)
#
#
#
#
# def preprocess_query(query):
#     query = RemovePunctuationsProcessor(query).get_tokens()
#     query = WordTokenizeProcessor(query).get_tokens()
#     query = SpellCheckerProcessor(query).get_tokens()
#     query = RemoveStopwordsProcessor(query).get_tokens()
#     query = LemmatizerProcessor(query).get_tokens()
#     query = StemmerProcessor(query).get_tokens()
#     return ' '.join(query)
#
# def preprocess_and_vectorize_query(query, vectorizer):
#     # Preprocess the query similarly to how documents were processed
#     processed_query = preprocess_query(query)  # Assume this function exists
#     query_vector = vectorizer.transform([processed_query])
#     return query_vector
#
#
# def load_index(file_name):
#     with open(file_name, 'rb') as f:
#         return pickle.load(f)


# inverted_index = load_index('inverted_index_csv.pkl')


# print(inverted_index)


# def search_query_with_cosine(query, inverted_index, document_vectors, document_count):
#     from collections import Counter
#     query_terms = preprocess_query(query).split()
#     query_vector = np.zeros(len(document_vectors[0]))  # Assuming document_vectors is a list of numpy arrays
#
#     # Construct the query vector
#     for term in query_terms:
#         if term in inverted_index:
#             term_index = list(inverted_index.keys()).index(term)
#             query_vector[term_index] = 1  # Or use actual TF-IDF values if available
#
#     # Calculate cosine similarity
#     relevant_documents = {}
#     for doc_id, doc_vector in enumerate(document_vectors):
#         if np.linalg.norm(doc_vector) * np.linalg.norm(query_vector):
#             cos_sim = np.dot(query_vector, doc_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(doc_vector))
#             relevant_documents[doc_id] = cos_sim
#
#     # Sort documents by the cosine similarity in descending order
#     sorted_docs = sorted(relevant_documents.items(), key=lambda x: x[1], reverse=True)
#     return sorted_docs[:10]


# def search_query(query, inverted_index):
#     processed_query = preprocess_query(query)
#     dd(processed_query)
#     scores = {}
#
#     # Collect scores from each term in the query
#     for term in processed_query:
#         if term in inverted_index:
#             for doc_id, tfidf_score in inverted_index[term].items():
#                 if doc_id in scores:
#                     scores[doc_id] += tfidf_score
#                 else:
#                     scores[doc_id] = tfidf_score
#
#     # Sort documents by their accumulated scores in descending order
#     sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
#     return sorted_docs[:10]  # Returning top 10 documents

# query = "girl"
# top_documents = search_query(query, inverted_index)
#
# # Print the results
# print("Top matching documents based on the query:")
# for doc_id, score in top_documents:
#     print(f"Document ID: {doc_id}, Score: {score}")

# def vectorize_query(query):
#     preprocessed_query = preprocess_query(query)
#     # print(query)
#     # quit()
#     query_vector = tfidf_vectorizer.transform([preprocessed_query])  # Note: transform expects an iterable
#     return query_vector
#
#
# def search_documents(query_vector, tfidf_matrix):
#     from sklearn.metrics.pairwise import cosine_similarity
#     # Calculate cosine similarity between the query vector and all document vectors
#     similarities = cosine_similarity(query_vector, tfidf_matrix)
#     # Sort document indices by similarity scores in descending order
#     top_indices = similarities.argsort()[0][::-1]
#     return top_indices
#
#
# def print_top_documents(top_indices, data, num_results=10):
#     top_docs = data.iloc[top_indices[:num_results]]
#     for index, row in top_docs.iterrows():
#         print(f"Document ID: {row['id_right']}, Content: {row['text_right']}\n")
#
#
# # Example query
# user_query = "columbia"
#
# # Vectorize the query
# query_vector = vectorize_query(user_query)
#
# # Search documents
# top_document_indices = search_documents(query_vector, tfidf_matrix)
# print("Top document indices:", print_top_documents(top_document_indices,full_data)[:10])  # Show top 10 results
