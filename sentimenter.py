import os

import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from dataclasses import dataclass
from nltk.tokenize import word_tokenize
import text_helpers
import pickle


@dataclass
class Corpus(object):
    word_index_map: np.ndarray
    document_matrix: np.ndarray


class Sentimenter:

    def __init__(self, stemmer, stopwords, labelled_docs, classifier):
        self.stemmer = stemmer
        self.stopwords = stopwords
        self.labelled_docs = labelled_docs
        self.corpus = self.get_corpus()
        self.model = classifier

    def tokenise(self, text):
        text = text_helpers.clean_text(text)
        word_tokens = word_tokenize(text)
        word_tokens = [t for t in word_tokens if len(t) > 2]

        word_tokens = [self.stemmer.stem(t) for t in word_tokens]
        word_tokens = [t for t in word_tokens if t not in self.stopwords and not t.isdigit()]
        return word_tokens

    def create_vocabulary(self):
        word_index_map = {}
        map_current_index = 0
        all_documents_with_tokens = []
        for doc in self.labelled_docs:
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
            # vectorise
            vector = np.zeros(len(word_index_map) + 1)
            for token in doc_tokens:
                vector[word_index_map[token]] += 1
            vector = vector / vector.sum()  # normalise
            vector[-1] = -1 if doc.label == "0" else 1
            document_matrix[i, :] = vector
            i += 1
        corpus = Corpus(word_index_map=word_index_map, document_matrix=document_matrix)
        return corpus

    def train(self):
        # Train Test Split
        X_train, X_test, Y_train, Y_test = train_test_split(self.corpus.document_matrix[:, :-1], self.corpus.document_matrix[:, -1], test_size=0.2, random_state=7)
        fold = KFold(n_splits=10, random_state=7)
        cv_results = cross_val_score(estimator=self.model, X=X_train, y=Y_train, cv=fold, scoring='accuracy', error_score='raise')
        print('{}: {} ({})'.format(self.model, cv_results.mean(), cv_results.std()))
        self.model.fit(X_train, Y_train)
        print("Train accuracy:", self.model.score(X_train, Y_train))
        print("Test accuracy:", self.model.score(X_test, Y_test))
        '''
        plt.show(block=True)
        plt.figure()
        plt.style.use('Solarize_Light2')
        fig = plt.figure()
        fig.suptitle('Logistic Regression')
        plt.boxplot(cv_results)
        plt.show()
        '''

    def get_corpus(self) -> Corpus:
        corpus_file = os.path.join('domain', 'twitter_corpus.pickle')
        if os.path.isfile(corpus_file):
            in_file = open(corpus_file, "rb")
            corpus = pickle.load(in_file)
            in_file.close()
            return corpus
        else:
            corpus = self.create_corpus()
            out_file = open(corpus_file, "wb")
            pickle.dump(corpus, out_file)
            out_file.close()
            return corpus
