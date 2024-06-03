import pandas as pd
from text_processors.remove_punctuations_processor import RemovePunctuationsProcessor
from text_processors.remove_stopwords_processor import RemoveStopwordsProcessor
from text_processors.stemmer_processor import StemmerProcessor
from text_processors.word_tokenize_processor import WordTokenizeProcessor
from text_processors.lemmatizer_processor import LemmatizerProcessor
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict


class CsvDataset:
    def load_csv(file_path):
        df = pd.read_csv(file_path, usecols=['id_right', 'text_right'], nrows=1000)
        df.rename(columns={'id_right': 'doc_id', 'text_right': 'text'}, inplace=True)
        df.index = ['Document'] * len(df)

        return df
    
    def csv_processing(text):
        text = RemovePunctuationsProcessor(text).get_tokens()
        text = WordTokenizeProcessor(text).get_tokens()
        text = RemoveStopwordsProcessor(text).get_tokens()
        text = LemmatizerProcessor(text).get_tokens()
        text = StemmerProcessor(text).get_tokens()

        return ' '.join(text)
    
    def storing_indexing_terms():
        global indexing_terms_df
        indexing_terms = set()

        for doc in csv_data.values:
            indexing_terms.update(WordTokenizeProcessor(doc[1]).tokens)
            indexing_terms_df = pd.DataFrame(indexing_terms)
        return indexing_terms_df
    
    def create_tfidf_features(data):
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(data['processed'])
        return vectorizer, X, data
    
    def build_inverted_index():
        inverted_index = defaultdict(list)

        for docId, doc in csv_data.values:
            doc_terms = set(WordTokenizeProcessor(doc).tokens)
            for term in doc_terms:
                inverted_index[term].append(docId)
        return dict(inverted_index)
    
    def calculate_tf_idf():
        documents = csv_data['text'].values
        vectorizer = TfidfVectorizer()
        # Fit the vectorizer to the documents
        tfidf_matrix = vectorizer.fit_transform(documents)
        df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out(), index=csv_data['doc_id'].values)
        return df