from random import sample
from typing import List
from domain.twitter_documents import Tweet
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns


class ScoreExplorer:
    
    def __init__(self, name: str, tweets: List[Tweet], predictions):
        self.name = name
        self.y_pred = predictions
        self.tweets_with_pred = list(zip(tweets, predictions))
        self.y_true = [t.label for t in tweets]

    def explore(self):
        self.see_classification_report()
        self.plot_multiclass_roc()
        self.show_random_correct_examples()
        self.show_random_misclassified_examples()

    def see_classification_report(self):
        print(self.name)
        print(classification_report(self.y_true, self.y_pred, labels=[-1, 0, 1], target_names=['negative', 'neutral', 'positive']))

    def show_random_correct_examples(self):
        print('Correctly classified')
        correctly_classified = list(filter(lambda t: t[0].label == t[1], self.tweets_with_pred))
        for t in sample(correctly_classified, 10):
            print('Tweet: {0} correctly predicted as: {1}'.format(t[0].text, t[1]))

    def show_random_misclassified_examples(self):
        misclassified = list(filter(lambda t: t[0].label != t[1], self.tweets_with_pred))
        for t in sample(misclassified, 10):
            print('Tweet: {0} predicted as: {1} instead of: {2}'.format(t[0].text, t[1], t[0].label))

    def plot_multiclass_roc(self, figsize=(17, 6)):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        sentiment_classes = [-1, 0, 1]
        sentiment_class_names = ['Negative', 'Neutral', 'Positive']

        for i in range(len(sentiment_classes)):
            sentiment = sentiment_classes[i]
            sentiment_true = [1 if t[0].label == sentiment else 0 for t in self.tweets_with_pred]
            sentiment_predict = [1 if t[1] == sentiment else 0 for t in self.tweets_with_pred]
            fpr[i], tpr[i], _ = roc_curve(sentiment_true, sentiment_predict)
            roc_auc[i] = auc(fpr[i], tpr[i])

        # roc for each class
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver operating characteristic')
        for i in range(len(sentiment_classes)):
            ax.plot(fpr[i], tpr[i], label='ROC curve (area = {0}) for Sentiment {1}'.format(roc_auc[i], sentiment_class_names[i]))
        ax.legend(loc="best")
        ax.grid(alpha=.4)
        sns.despine()
        plt.show()


