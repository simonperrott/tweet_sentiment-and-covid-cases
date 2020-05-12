import collections
import os
import time

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import pickle
import glob

from domain.corpus import CorpusManager


class SentimentAnalyser:

    def __init__(self, corpus_manager: CorpusManager, classifier):
        self.corpus_manager = corpus_manager
        model_name = type(classifier).__name__
        trained_model = self.load_last_model(model_name)
        if trained_model:
            self.model = trained_model
        else:
            self.model = classifier
            self.__train()
            self.__save_model(model_name)

    def __train(self):
        document_matrix = self.corpus_manager.corpus.document_matrix
        X_train, X_test, y_train, y_test = train_test_split(document_matrix[:, :-1], document_matrix[:, -1], test_size=0.2, random_state=7, shuffle=True)

        # Tune Hyperparams
        self.model = GridSearchCV(self.model, param_grid=[{'C': [0.001,0.01,0.1,1,10,100]}], cv=5)
        self.model.fit(X_train, y_train)
        print(self.model.best_params_)
        for param, score in zip(self.model.cv_results_['params'], self.model.cv_results_['mean_test_score']):
            print(param, score)

        # fold = KFold(n_splits=7)
        # cv_results = cross_val_score(estimator=self.model, X=X_train, y=y_train, cv=fold, scoring='accuracy', error_score='raise')
        # print('{}: {} ({})'.format(self.model, cv_results.mean(), cv_results.std()))

        print("Train accuracy:", self.model.score(X_train, y_train))
        print("Test accuracy:", self.model.score(X_test, y_test))
        print('Baseline: Train accuracy = 0.7244292793890649 & Test accuracy = 0.6692657569850552')
        self.__plot_confusion_matrix(X_test, y_test)

    def classify(self, documents):
        all_doc_tokens = [self.corpus_manager.tokenise(doc.text) for doc in documents]
        vectors = [self.corpus_manager.vectorise(doc_tokens) for doc_tokens in all_doc_tokens]
        predictions = self.model.predict(vectors)
        for i in range(len(documents)):
            documents[i].label = predictions[i]
        self.corpus_manager.show_sentiment_wordclouds(list(zip(documents, all_doc_tokens)))
        return documents

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

    def load_last_model(self, model_name):
        list_of_model_files = glob.glob(str.format('domain/models/{0}_*', model_name))
        if len(list_of_model_files) > 0:
            latest_file = max(list_of_model_files, key=os.path.getctime)
            if os.path.isfile(latest_file):
                in_file = open(latest_file, "rb")
                model = pickle.load(in_file)
                in_file.close()
                return model
            else:
                return None

    def plot_model_comparison(self):
        pass
