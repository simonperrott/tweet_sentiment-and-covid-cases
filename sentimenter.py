import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score

from nltk.tokenize import word_tokenize
import text_helpers


class Sentimenter:

    def __init__(self, stemmer, stopwords, labelled_docs, classifier):
        self.stemmer = stemmer
        self.stopwords = stopwords
        self.word_index_map, self.documents_tokenized = self.__create_corpus_vocabulary(labelled_docs)
        self.training_data_matrix = self.__create_matrix()
        self.model = classifier

    def __tokenise(self, text):
        text = text_helpers.clean_text(text)
        word_tokens = word_tokenize(text)
        word_tokens = [t for t in word_tokens if len(t) > 2]
        word_tokens = [self.stemmer.stem(t) for t in word_tokens]
        word_tokens = [t for t in word_tokens if t not in self.stopwords]
        return word_tokens

    def __create_corpus_vocabulary(self, docs):
        word_index_map = {}
        map_current_index = 0
        all_documents_with_tokens = []
        for doc in docs:
            tokens = self.__tokenise(doc.text)
            all_documents_with_tokens.append((doc, tokens))
            for t in tokens:
                if t not in self.word_index_map:
                    self.word_index_map[t] = map_current_index
                    map_current_index += 1
        print("length of corpus vocabulary:", len(self.word_index_map))
        return word_index_map, all_documents_with_tokens

    def __vectorise(self, tokens, label):
        x = np.zeros(len(self.word_index_map) + 1)
        for t in tokens:
            x[self.word_index_map[t]] += 1
        x = x / x.sum()  # normalise
        x[-1] = label
        return x

    def __create_matrix(self):
        N = len(self.documents_tokenized)
        data = np.zeros((N, len(self.word_index_map) + 1)) # one document in each row with its label in the last column
        i = 0
        for doc, doc_tokens in self.documents_tokenized:
            xy = self.__vectorise(doc_tokens, doc.label)
            data[i, :] = xy
            i += 1
        return data

    def train(self):
        # Train Test Split
        X_train, X_test, Y_train, Y_test = train_test_split(self.training_data_matrix[:, :-1], self.training_data_matrix[:, -1], test_size=0.2, random_state=7)
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