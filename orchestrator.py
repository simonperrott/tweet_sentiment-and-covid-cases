import random
from typing import List

import datetime
import dateutil
import numpy as np
import pandas as pd

from domain import covid_timeseries
from domain.covid_timeseries import load_covid_deaths
from domain.twitter_documents import AirlineTweetsManager, SemEvalTweetsManager, LeaderTweetsManager, TwitterApiManager, \
    Tweet
from sentimenter import CustomSentimentAnalyser


def start():
    set_seed(10)

    training_data: List[Tweet] = load_training_data()
    random.shuffle(training_data)
    ultimate_test_set = training_data[:100]
    training_set = training_data[100:]

    '''
    # nltk Vader is already trained so we'll use the ultimate test data to explore its performance
    vader_sentiment_analyser = VaderSentimentAnalyser(0.3)
    vader_predicted_labels = vader_sentiment_analyser.classify(ultimate_test_set)
    scorer = ScoreExplorer('Vader Sentiment Classification', ultimate_test_set, vader_predicted_labels)
    scorer.explore()
    '''

    # my own custom sentiment analyser built from the ground up needs training
    # not using the same ultimate test data in training so we can use the same test data to evaluate as we used for vader.
    sentimenter_analyser = CustomSentimentAnalyser(training_set)
    '''
    sentimenter_analyser.train()
    custom_predicted_labels = sentimenter_analyser.classify(ultimate_test_set)
    scorer = ScoreExplorer('Custom Sentiment Classification', ultimate_test_set, custom_predicted_labels)
    scorer.explore()
    '''

    # Load Covid Data
    covid_cases = covid_timeseries.load_covid_cases()
    covid_deaths = covid_timeseries.load_covid_deaths()

    # classify leader tweets
    tweets = load_leader_tweets(True)
    predicted_labels = sentimenter_analyser.classify(tweets)
    for idx, tweet in enumerate(tweets):
        tweet.label = predicted_labels[idx]
    tweets_df = pd.DataFrame([t.to_dict() for t in tweets])
    trump_daily_sentiment_average = pd.Series
    tweet_grp_sentiment_means = tweets_df.groupby(['author', 'date'])['label'].mean()
    for name, val in tweet_grp_sentiment_means:
        if('Trump' in name):
            trump_daily_sentiment_average.set_value(name, val)

    # for t in tweets if t.author == '@realDonaldTrump'])
    # group tweets by day with average rounded sentiment for that day

    # uk_tweets_df = pd.DataFrame([t.to_dict() for t in tweets if t.author == '@BorisJohnson'])
    # irish_tweets_df = pd.DataFrame([t.to_dict() for t in tweets if t.author == '@LeoVaradkar'])

    explore_correlations()


def set_seed(args):
    random.seed(args)
    np.random.seed(args)


def load_leader_tweets(update_with_latest=False):
    api_mgr = TwitterApiManager()
    leader_tweets_mgr = LeaderTweetsManager()
    leader_tweets: List[Tweet] = leader_tweets_mgr.load_documents()
    if update_with_latest:
        new_tweets = api_mgr.get_more_tweets(leader_tweets)
        if len(new_tweets) > 0:
            leader_tweets_mgr.save_documents('a', new_tweets)
            leader_tweets.extend(new_tweets)
    return list(filter(lambda t: dateutil.parser.parse(t.date) > datetime.datetime(2020, 1, 22),  leader_tweets))


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
    return list(training_tweets_balanced.itertuples())


def explore_correlations():
    deaths = load_covid_deaths()
    ireland_deaths: pd.Series = deaths['Ireland']
    uk_deaths: pd.Series = deaths['United Kingdom']
    us_deaths: pd.Series = deaths['US']
    days = deaths['days']
    # for day in days:
        # av sentiment on that day


start()
