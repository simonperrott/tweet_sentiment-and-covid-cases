import random
from typing import List

import datetime
import dateutil
import numpy as np
import pandas as pd

from domain import covid_timeseries
from domain.twitter_documents import AirlineTweetsManager, SemEvalTweetsManager, LeaderTweetsManager, TwitterApiManager, \
    Tweet
from sentimenter import CustomSentimentAnalyser
import seaborn as sn
import matplotlib.pyplot as plt


def set_seed(args):
    random.seed(args)
    np.random.seed(args)


def load_training_data():
    airline_tweets = AirlineTweetsManager().load_documents()
    semeval_tweets = SemEvalTweetsManager().load_documents()
    labelled_leader_tweets = [tweet for tweet in LeaderTweetsManager('labelled_leader_tweets.csv').load_documents() if tweet.label]
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


def load_tweets_to_classify(update_with_latest=False):
    api_mgr = TwitterApiManager()
    leader_tweets_mgr = LeaderTweetsManager()
    leader_tweets: List[Tweet] = leader_tweets_mgr.load_documents()
    if update_with_latest:
        new_tweets: List[Tweet] = api_mgr.get_more_tweets(leader_tweets)
        if len(new_tweets) > 0:
            leader_tweets_mgr.save_documents('a', [t.to_dict() for t in new_tweets])
            leader_tweets.extend(new_tweets)
    return list(filter(lambda t: dateutil.parser.parse(t.date) > datetime.datetime(2020, 1, 22),  leader_tweets))


class Orchestrator:

    def __init__(self):
        self.covid_cases = covid_timeseries.load_covid_cases()
        self.covid_deaths = covid_timeseries.load_covid_deaths()

    def start(self):
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

        # classify leader tweets
        tweets = load_tweets_to_classify(True)
        predicted_labels = sentimenter_analyser.classify(tweets)
        for idx, tweet in enumerate(tweets):
            tweet.label = predicted_labels[idx]

        country_dfs = self.create_country_dataframes(tweets)
        for country_df in country_dfs:
            print(country_df.name)
            corr_matrix = country_df.corr()
            print(corr_matrix)
            plt.subplots(figsize=(18, 6))
            plt.title(country_df.name)
            sn.heatmap(corr_matrix, annot=True)
            plt.show()

    def create_country_dataframes(self, tweets: List[Tweet]):
        leader_to_country_lookup = {'@realDonaldTrump': 'US', '@LeoVaradkar': 'Ireland', '@BorisJohnson': 'United Kingdom'}

        # Create a dataframe with all tweets until our chosen date and index by calendar day.
        tweets_df = pd.DataFrame([t.to_dict() for t in tweets])
        tweets_df['date'] = tweets_df['date'].apply(lambda d: dateutil.parser.parse(d))
        tweets_df = tweets_df[tweets_df['date'] <= dateutil.parser.parse(self.covid_cases.index.max())]
        tweets_df['day'] = tweets_df['date'].apply(lambda d: d.strftime("%Y-%m-%d"))

        # Group tweets per calendar day and per leader to calculate the number of tweets as well as the average sentiment for each leader on a day
        grouped = tweets_df.groupby(['author', 'day'])
        df = grouped.agg(num_tweets=pd.NamedAgg(column='label', aggfunc='count'), av_sentiment=pd.NamedAgg(column='label', aggfunc='mean'))
        df['leader'] = df.index.get_level_values(0)
        df['country'] = df['leader'].apply(lambda x: leader_to_country_lookup[x])

        # Combine with covid cases and deaths
        country_dfs = []
        for country in leader_to_country_lookup.values():
            country_df = df[df['country'] == country].reset_index('author', drop=True) # dropping the author part of the multilevel index
            country_df['cases'] = self.covid_cases[country].astype(int)
            country_df['deaths'] = self.covid_deaths[country].astype(int)
            country_df.name = country
            country_dfs.append(country_df)
        return country_dfs


Orchestrator().start()
