import os
import pickle
from dataclasses import dataclass

import numpy as np
from nltk import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import text_helpers
import collections


@dataclass
class Corpus(object):
    word_index_map: np.ndarray
    document_matrix: np.ndarray
    inverse_doc_freq: dict


class CorpusManager:

    flight_stopwords = ['country', 'flight', 'JetBlue', 'USAirways', 'SouthwestAir', 'AmericanAir', 'VirginAmerica', 'luggage', 'bag', 'flying', 'flight', 'fly', 'agent', 'amp', 'plane', 'airline', 'airport', 'seat', 'gate']
    tech_stopwords = ['iphone', 'amazon', 'prime', 'apple', 'google']

    def __init__(self, training_docs, stemmer, stopwords, min_word_length):
        self.min_word_length = min_word_length
        self.corpus_file = os.path.join('domain', 'twitter_corpus.pickle')
        self.stemmer = stemmer
        self.stopwords =  stopwords.union(CorpusManager.flight_stopwords).union(CorpusManager.tech_stopwords)
        self.corpus = self.load_corpus() or self.create_corpus(training_docs)

    def __tokenise(self, text):
        text = text_helpers.clean_text(text)

        word_tokens = word_tokenize(text)
        word_tokens = [t for t in word_tokens if len(t) > self.min_word_length]
        word_tokens = [t for t in word_tokens if t not in self.stopwords and not t.isdigit()]

        word_tokens = [self.stemmer.stem(t) for t in word_tokens]

        return word_tokens

    def __create_vocabulary(self, training_docs):
        word_index_map = {}
        map_current_index = 0
        all_documents_with_tokens = []
        for row in training_docs.itertuples():
            tokens = self.__tokenise(row.text)
            if len(tokens) > 0:
                all_documents_with_tokens.append((row, tokens))
                for t in tokens:
                    if t not in word_index_map:
                        word_index_map[t] = map_current_index
                        map_current_index += 1
        print("length of corpus vocabulary:", len(word_index_map))
        return word_index_map, all_documents_with_tokens

    def load_corpus(self) -> Corpus:
        if os.path.isfile(self.corpus_file):
            in_file = open(self.corpus_file, "rb")
            corpus = pickle.load(in_file)
            in_file.close()
            return corpus
        else:
            return None

    def create_corpus(self, training_docs):
        word_index_map, documents_tokenized = self.__create_vocabulary(training_docs)
        num_of_docs = len(documents_tokenized)
        document_matrix = np.zeros((num_of_docs, len(word_index_map) + 1))  # one document in each row with its label in the last column
        i = 0
        all_documents_all_tokens = [d[1] for d in documents_tokenized]
        dict_of_idf = {}
        for doc, doc_tokens in documents_tokenized:
            vector = np.zeros(len(word_index_map) + 1)
            for token in set(doc_tokens):
                term_factor = doc_tokens.count(token)
                if token in dict_of_idf:
                    inverse_document_freq_log = dict_of_idf[token]
                else:
                    count_of_docs_having_token = sum(map(lambda x: token in x, all_documents_all_tokens))
                    inverse_document_freq_log = np.log(num_of_docs / count_of_docs_having_token)
                    dict_of_idf[token] = inverse_document_freq_log
                vector[word_index_map[token]] = term_factor * inverse_document_freq_log
            vector[-1] = doc.label
            document_matrix[i, :] = vector
            i += 1
        corpus = Corpus(word_index_map=word_index_map, document_matrix=document_matrix, inverse_doc_freq=dict_of_idf)
        # self.show_sentiment_wordclouds(documents_tokenized)
        self.save_corpus(corpus)
        return corpus

    def save_corpus(self, corpus):
        out_file = open(self.corpus_file, "wb")
        pickle.dump(corpus, out_file)
        out_file.close()

    def show_sentiment_wordclouds(self, documents_tokenized):
        for sentiment in [(-1.0, 'negative'), (0.0, 'neutral'), (1.0, 'positive')]:
            list_of_docs = [' '.join(doc[1]) for doc in documents_tokenized if doc[0].label == sentiment[0]]
            all_tokens = ' '.join(list_of_docs)
            most_common_tokens = collections.Counter(all_tokens.split()).most_common(40)
            self.__plot_word_clouds(most_common_tokens, sentiment[1])

    def vectorise(self, documents):
        all_doc_tokens = [self.__tokenise(doc.text) for doc in documents]
        vectors = [self._vectorise_single_doc(doc_tokens) for doc_tokens in all_doc_tokens]
        return vectors

    def _vectorise_single_doc(self, doc_tokens):
        vector = np.zeros(len(self.corpus.word_index_map))
        for token in set(doc_tokens):
            if token in self.corpus.word_index_map and token in self.corpus.inverse_doc_freq:
                term_factor = doc_tokens.count(token)
                inverse_doc_factor = self.corpus.inverse_doc_freq[token]
                vector[self.corpus.word_index_map[token]] = term_factor * inverse_doc_factor
        return vector

    @staticmethod
    def __plot_word_clouds(words, sentiment):
        wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate_from_frequencies(dict(words))
        plt.figure(figsize=(10, 7))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis('off')
        plt.title('{0} sentiment'.format(sentiment))
        plt.show()
