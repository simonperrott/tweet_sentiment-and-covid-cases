import os

import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from nltk.tokenize import word_tokenize
import text_helpers
import pickle

from domain.corpus import Corpus


class Sentimenter:

    def __init__(self, stemmer, stopwords, labelled_docs, classifier):
        self.stemmer = stemmer
        self.stopwords = stopwords
        self.labelled_docs = labelled_docs
        self.corpus = self.get_corpus()
        self.model = classifier

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


