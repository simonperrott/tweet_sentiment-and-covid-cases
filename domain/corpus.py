import os
import pickle
from dataclasses import dataclass

import numpy as np
from nltk import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import text_helpers


@dataclass
class Corpus(object):
    word_index_map: np.ndarray
    document_matrix: np.ndarray
    all_documents_with_tokens: np.array


class CorpusManager:

    flight_stopwords = ['country', 'flight', 'JetBlue', 'USAirways', 'SouthwestAir', 'AmericanAir', 'VirginAmerica', 'luggage', 'bag', 'flying', 'flight', 'fly', 'agent', 'amp', 'plane', 'airline', 'airport', 'seat', 'gate']
    tech_stopwords = ['iphone', 'amazon', 'prime', 'apple', 'google']

    def __init__(self, training_docs, stemmer, stopwords):
        self.corpus_file = os.path.join('domain', 'twitter_corpus.pickle')
        self.stemmer = stemmer
        self.stopwords = stopwords.union(CorpusManager.flight_stopwords).union(CorpusManager.tech_stopwords)
        self.training_documents = training_docs
        self.corpus = self.load_corpus() or self.create_corpus()

    def load_corpus(self) -> Corpus:
        if os.path.isfile(self.corpus_file):
            in_file = open(self.corpus_file, "rb")
            corpus = pickle.load(in_file)
            in_file.close()
            return corpus
        else:
            return None

    def tokenise(self, text):
        text = text_helpers.clean_text(text)

        word_tokens = word_tokenize(text)
        word_tokens = [t for t in word_tokens if len(t) > 2]
        word_tokens = [t for t in word_tokens if t not in self.stopwords and not t.isdigit()]

        word_tokens = [self.stemmer.stem(t) for t in word_tokens]

        return word_tokens

    def create_vocabulary(self):
        word_index_map = {}
        map_current_index = 0
        all_documents_with_tokens = []
        for doc in self.training_documents:
            tokens = self.tokenise(doc.text)
            all_documents_with_tokens.append((doc, tokens))
            for t in tokens:
                if t not in word_index_map:
                    word_index_map[t] = map_current_index
                    map_current_index += 1
        print("length of corpus vocabulary:", len(word_index_map))
        return word_index_map, all_documents_with_tokens

    def create_corpus(self):
        word_index_map, documents_tokenized = self.create_vocabulary()
        N = len(documents_tokenized)
        document_matrix = np.zeros((N, len(word_index_map) + 1))  # one document in each row with its label in the last column
        i = 0
        for doc, doc_tokens in documents_tokenized:
            if len(doc_tokens) > 0:
                # vectorise
                vector = np.zeros(len(word_index_map) + 1)
                for token in doc_tokens:
                    vector[word_index_map[token]] += 1
                vector = vector / vector.sum()  # normalise
                vector[-1] = doc.sentiment
                document_matrix[i, :] = vector
                i += 1
        corpus = Corpus(word_index_map=word_index_map, document_matrix=document_matrix, all_documents_with_tokens=documents_tokenized)
        self.save_corpus(corpus)
        self.__show_corpus_sentiment_wordclouds(documents_tokenized)
        return corpus

    def __show_corpus_sentiment_wordclouds(self, documents_tokenized):
        for sentiment in [('-1', 'negative'), ('0', 'neutral'), ('1', 'positive')]:
            list_of_doc_tokens = [' '.join(doc[1]) for doc in documents_tokenized if doc[0].sentiment == sentiment[0]]
            words = ' '.join(list_of_doc_tokens)
            self.__plot_word_clouds(words, sentiment[1])

    def vectorise(self, tokens):
        word_index_map = self.corpus.word_index_map
        vector = np.zeros(len(word_index_map))
        for token in tokens:
            if token in word_index_map:
                vector[word_index_map[token]] += 1
        if vector.sum() > 0:
            vector = vector / vector.sum()  # normalise
        return vector

    def save_corpus(self, corpus):
        out_file = open(self.corpus_file, "wb")
        pickle.dump(corpus, out_file)
        out_file.close()

    @staticmethod
    def __plot_word_clouds(words, sentiment):
        wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(words)
        plt.figure(figsize=(10, 7))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis('off')
        plt.title('{0} sentiment'.format(sentiment))
        plt.show()
