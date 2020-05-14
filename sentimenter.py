import os
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import plot_confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

import pickle
import glob

from domain.corpus import CorpusManager


class SentimentAnalyser:

    def __init__(self):
        self.model = self._load_last_model()

    def train(self, document_matrix):
        X_train, X_test, y_train, y_test = train_test_split(document_matrix[:, :-1], document_matrix[:, -1], test_size=0.3, random_state=7, shuffle=True)

        # Train and compare multipole models whilst tuning their hyperparameters using a pipeline
        pipeline = Pipeline([("classifier", LogisticRegression())])

        search_space = [{'classifier': [LogisticRegression()],
                         'classifier__C': [0.0001, 0.001],
                         'classifier__solver': ['liblinear', 'lbfgs']},
                        {'classifier': [RandomForestClassifier()],
                         'classifier__n_estimators': [10, 100, 1000],
                         'classifier__max_features': [3, 5, 10, 100]},
                        {'classifier': [MultinomialNB()],
                         'classifier__alpha': [1, 1e-1, 1e-2]}
                        ]

        model = RandomizedSearchCV(pipeline, search_space, cv=5, n_jobs=-1)
        # n_jobs parameter = -1, grid search will detect how many cores are installed and use them all
        model.fit(X_train, y_train)
        self.model = model

        # View best model
        best_model = model.best_estimator_.get_params()['classifier']
        print(best_model)

        print(model.best_params_)
        for param, score in zip(model.cv_results_['params'], model.cv_results_['mean_test_score']):
            print(param, score)

        # fold = KFold(n_splits=7)
        # cv_results = cross_val_score(estimator=self.model, X=X_train, y=y_train, cv=fold, scoring='accuracy', error_score='raise')
        # print('{}: {} ({})'.format(self.model, cv_results.mean(), cv_results.std()))

        print("Train accuracy:", model.score(X_train, y_train))
        print("Test accuracy:", model.score(X_test, y_test))
        print('Baseline: Train accuracy = 0.7244292793890649 & Test accuracy = 0.6692657569850552')
        self.__plot_confusion_matrix(X_test, y_test)
        self.__save_model(type(best_model).__name__)

    def classify(self, vectors):
        predictions = self.model.predict(vectors)
        return predictions

    def __plot_confusion_matrix(self, X_test, y_test):
        disp = plot_confusion_matrix(self.model, X_test, y_test,
                                     cmap=plt.cm.Blues,
                                     normalize='true')
        disp.ax_.set_title("Normalized confusion matrix")
        plt.show()

    def __save_model(self, model_name):
        file_name = str.format('{0}_{1}.pickle', model_name, round(time.time()))
        qualified_file_name = os.path.join('domain/models', file_name)
        out_file = open(qualified_file_name, "wb")
        pickle.dump(self.model, out_file)
        out_file.close()

    def _load_last_model(self):
        list_of_model_files = glob.glob('domain/models/*')
        if len(list_of_model_files) > 0:
            latest_file = max(list_of_model_files, key=os.path.getctime)
            if os.path.isfile(latest_file):
                in_file = open(latest_file, "rb")
                model = pickle.load(in_file)
                in_file.close()
                return model
            else:
                return None