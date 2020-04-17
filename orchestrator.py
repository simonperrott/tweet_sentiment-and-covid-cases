from sentimenter import Sentimenter
from domain.twitter_documents import TweetManager, TrainingDataManager
import random
import numpy as np
from nltk.stem import SnowballStemmer
from sklearn.linear_model import LogisticRegression


def start():
    # set_seed(10)
    # update_leader_tweets()
    stop_words = []  # set(w.rstrip() for w in open('data/stopwords.txt'))
    training_data = get_training_data()
    sentimenter = Sentimenter(SnowballStemmer(language='english'), stop_words, training_data, LogisticRegression())
    sentimenter.train()
    # save model

def set_seed(args):
    random.seed(args)
    np.random.seed(args)


def update_leader_tweets():
    mgr = TweetManager()
    authors = ["@BorisJohnson", "@LeoVaradkar", "@realDonaldTrump"]
    new_tweets = mgr.get_more_tweets(authors)
    if len(new_tweets) > 0:
        mgr.save_documents('a', new_tweets)

def get_training_data():
    mgr = TrainingDataManager()
    training_tweets = mgr.load_documents()
    return training_tweets[0:100]  # TODO: Remove when code proven to have full dataset


start()
