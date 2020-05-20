from abc import ABC
from collections import namedtuple
from typing import List

import twitter

from domain.base_document import DocumentManager
from numpy import array
from datetime import datetime, timezone
import dateutil.parser
import dateutil.tz
from configparser import ConfigParser


class Tweet:

    def __init__(self, tweet_id, text, label, author=None, date=None):
        self.label = self.__parse_label(label)
        self.text = text
        self.date = date
        self.author = author
        self.tweet_id = tweet_id

    @staticmethod
    def __parse_label(label):
        if label in ['positive', 4, '4']:
            return 1
        elif label in ['neutral', 2, '2']:
            return 0
        else:
            return -1

    def to_dict(self):
        return {
            "tweet_id":self.tweet_id,
            "text": self.text,
            "label": self.label,
            "author": self.author,
            "date": self.date
        }


class TwitterApiManager:

    def __init__(self):

        parser = ConfigParser()
        parser.read('twitter_api_config.ini')
        # initialize api instance
        self.twitter_api = twitter.Api(consumer_key=parser['twitter']['consumer_key'],
                                       consumer_secret=parser['twitter']['consumer_secret'],
                                       access_token_key=parser['twitter']['access_token_key'],
                                       access_token_secret=parser['twitter']['access_token_secret'],
                                       sleep_on_rate_limit=True)

        # test authentication
        print(self.twitter_api.VerifyCredentials())
        self.authors = ['@realDonaldTrump', '@LeoVaradkar', '@BorisJohnson']

    @staticmethod
    def reformat_date(date_string):
        #  date format = 'Fri Mar 06 20:42:39 +0000 2020'
        tweet_time = dateutil.parser.parse(date_string)
        # tweet_time = datetime.strptime(date, '%a %b %d %H:%M:%S %z %Y')
        return tweet_time.strftime("%Y/%m/%d, %H:%M:%S")

    def api_get_user_timeline(self, twitter_user, count=200, max_id: int = None, since_id=None) -> array:
        tweets_fetched = []
        while len(tweets_fetched) < count:
            try:
                if since_id:
                    if max_id:
                        response = self.twitter_api.GetUserTimeline(screen_name=twitter_user, trim_user=True, count=200, max_id=max_id-1, since_id=since_id)
                    else:
                        response = self.twitter_api.GetUserTimeline(screen_name=twitter_user, trim_user=True, count=200, since_id=since_id)
                else:
                    if max_id:
                        response = self.twitter_api.GetUserTimeline(screen_name=twitter_user, trim_user=True, count=200, max_id=max_id-1)
                    else:
                        response = self.twitter_api.GetUserTimeline(screen_name=twitter_user, trim_user=True, count=200)

                [tweets_fetched.append(Tweet(x.id, x.text, None, twitter_user, TwitterApiManager.reformat_date(x.created_at)))
                 for x in response if x.lang == 'en' and dateutil.parser.parse(x.created_at) >= datetime(2020, 1, 22, 0, 0, 0, tzinfo=timezone.utc)]

                if len(response) == 0:
                    break
                max_id = response[-1].id
            except Exception as e:
                    pass
        return tweets_fetched

    def get_more_tweets(self, existing_tweets: List[Tweet]):
        new_tweets = []
        for author in self.authors:
            since_id = None
            author_tweets = list(filter(lambda t: t.author == author, existing_tweets))
            if len(author_tweets) > 0:
                latest_tweet = max(author_tweets, key=lambda x: x.date)
                since_id = latest_tweet.tweet_id
            new_tweets.extend(self.api_get_user_timeline(author, count=5000, since_id=since_id))
        return new_tweets


class LeaderTweetsManager(DocumentManager):

    def __init__(self, headers, filename=None):
        if not filename:
            filename = 'leader_tweets.csv'
        super().__init__(directory='twitter', filename=filename, encoding='utf-8', headers=headers)

    def create_document(self, file_row):
        return Tweet(tweet_id=file_row[self.headers.index('tweet_id')],
                     text=file_row[self.headers.index('text')],
                     label=file_row[self.headers.index('label')],
                     author=file_row[self.headers.index('author')],
                     date=file_row[self.headers.index('date')])


class AirlineTweetsManager(DocumentManager):

    def __init__(self):
        super().__init__(directory='twitter', filename='airline_tweets.csv', encoding='latin-1', headers=['tweet_id', 'airline_sentiment', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'text'])

    def create_document(self, file_row):
        return Tweet(tweet_id=file_row[self.headers.index('tweet_id')],
                     text=file_row[self.headers.index('text')],
                     label=file_row[self.headers.index('airline_sentiment')])


class SemEvalTweetsManager(DocumentManager):

    def __init__(self):
        super().__init__(directory='twitter', filename='semeval_tweets_2016_taskA.csv', encoding='latin-1', separator='\t', headers=None)

    def create_document(self, file_row):
        return Tweet(tweet_id=file_row[0],
                     text=file_row[2],
                     label=file_row[1])