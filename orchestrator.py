from sentimenter import SentimentAnalyser
from domain.twitter_documents import AirlineTweetsManager, SemEvalTweetsManager, LeaderTweetsManager, TwitterApiManager
import random
import numpy as np
from nltk.stem import SnowballStemmer
from domain.corpus import CorpusManager
import pandas as pd


def start():
    set_seed(10)
    # update_leader_tweets()

    # Config how we build our corpus
    stop_words = set(w.rstrip() for w in open('data/stopwords.txt'))
    min_word_length = 2

    training_data: pd.DataFrame = load_training_data()
    corpus_mgr = CorpusManager(training_data, SnowballStemmer(language='english'), stop_words, min_word_length)

    sentimenter_analyser = SentimentAnalyser()
    sentimenter_analyser.train(corpus_mgr.corpus.document_matrix)

    tweets = load_leader_tweets(False)
    tweets_to_classify = random.sample(tweets, 1000)
    vectors, all_tweet_tokens = corpus_mgr.vectorise(tweets_to_classify)
    predicted_labels = sentimenter_analyser.classify(vectors)
    for i in range(len(tweets_to_classify)):
        tweets_to_classify[i].label = predicted_labels[i]
    corpus_mgr.show_sentiment_wordclouds(list(zip(tweets_to_classify, all_tweet_tokens)))


def set_seed(args):
    random.seed(args)
    np.random.seed(args)


def load_leader_tweets(update_with_latest=False):
    api_mgr = TwitterApiManager()
    leader_tweets_mgr = LeaderTweetsManager()
    leader_tweets = leader_tweets_mgr.load_documents()
    if update_with_latest:
        new_tweets = api_mgr.get_more_tweets(leader_tweets)
        if len(new_tweets) > 0:
            leader_tweets_mgr.save_documents('a', new_tweets)
            leader_tweets.extend(new_tweets)
    return leader_tweets


def load_training_data():
    airline_tweets = AirlineTweetsManager().load_documents()
    semeval_tweets = SemEvalTweetsManager().load_documents()
    labelled_leader_tweets = [tweet for tweet in LeaderTweetsManager().load_documents() if tweet.label]
    training_tweets = []
    training_tweets.extend(airline_tweets)
    training_tweets.extend(semeval_tweets)
    training_tweets.extend(labelled_leader_tweets)

    # Get even number of tweets for each sentiment
    df = pd.DataFrame([t.to_dict() for t in training_tweets])
    groups = df.groupby(df.label)
    min_count = min(groups.size())
    training_tweets_balanced = groups.apply(lambda x: x.sample(n=min_count))

    print('Training data: Total={0}, Positive={1}, Neutral={2}, Negative={3}'.format(len(training_tweets_balanced)
                                                                                     , len(training_tweets_balanced[training_tweets_balanced.label == 1])
                                                                                     , len(training_tweets_balanced[training_tweets_balanced.label == 0])
                                                                                     , len(training_tweets_balanced[training_tweets_balanced.label == -1])
                                                                                     ))
    return training_tweets_balanced


start()