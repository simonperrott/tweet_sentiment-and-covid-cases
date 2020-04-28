from sentimenter import SentimentAnalyser
from domain.twitter_documents import TweetManager, TrainingDataManager
import random
import numpy as np
from nltk.stem import SnowballStemmer
from sklearn.linear_model import LogisticRegression
from domain.corpus import CorpusManager


def start():
    # set_seed(10)
    # update_leader_tweets()
    stop_words = set(w.rstrip() for w in open('data/stopwords.txt'))
    training_data = load_training_data()
    corpus_mgr = CorpusManager(training_data, SnowballStemmer(language='english'), stop_words)
    sentimenter = SentimentAnalyser(corpus_mgr, LogisticRegression())
    tweets_to_classify = random.sample(load_leader_tweets(False), 1000)
    results = sentimenter.classify(tweets_to_classify)


def set_seed(args):
    random.seed(args)
    np.random.seed(args)


def load_leader_tweets(update_with_latest=False):
    mgr = TweetManager()
    authors = ["@BorisJohnson", "@LeoVaradkar", "@realDonaldTrump"]
    if update_with_latest:
        new_tweets = mgr.get_more_tweets(authors)
        if len(new_tweets) > 0:
            mgr.save_documents('a', new_tweets)
    leader_tweets = mgr.load_documents()
    return leader_tweets

def load_training_data():
    mgr = TrainingDataManager()
    training_tweets = mgr.load_documents()
    return training_tweets


start()
