from score_explorer import ScoreExplorer
from sentimenter import CustomSentimentAnalyser, VaderSentimentAnalyser
from domain.twitter_documents import AirlineTweetsManager, SemEvalTweetsManager, LeaderTweetsManager, TwitterApiManager, Tweet
import random
import numpy as np
import pandas as pd
from typing import List


def start():
    set_seed(10)

    training_data: List[Tweet] = load_training_data()
    random.shuffle(training_data)
    ultimate_test_set = training_data[:100]
    training_set = training_data[100:]

    # nltk Vader is already trained so we'll use the ultimate test data to explore its performance
    vader_sentiment_analyser = VaderSentimentAnalyser(0.3)
    vader_predicted_labels = vader_sentiment_analyser.classify(ultimate_test_set)
    scorer = ScoreExplorer('Vader Sentiment Classification', ultimate_test_set, vader_predicted_labels)
    scorer.explore()

    # my own custom sentiment analyser built from the ground up needs training
    # not using the same ultimate test data in training so we can use the same test data to evaluate as we used for vader.
    sentimenter_analyser = CustomSentimentAnalyser(training_set)
    sentimenter_analyser.train()
    custom_predicted_labels = sentimenter_analyser.classify(ultimate_test_set)
    scorer = ScoreExplorer('Custom Sentiment Classification', ultimate_test_set, custom_predicted_labels)
    scorer.explore()

    # plot a comparison of sentiment analysers


    # classify leader tweets
    tweets = load_leader_tweets(False)
    tweets_to_classify = random.sample(tweets, 1000)
    predicted_labels = sentimenter_analyser.classify(tweets)
    for i in range(len(tweets_to_classify)):
        tweets_to_classify[i].label = predicted_labels[i]

    # explore correlations


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
    return list(training_tweets_balanced.itertuples())


start()
