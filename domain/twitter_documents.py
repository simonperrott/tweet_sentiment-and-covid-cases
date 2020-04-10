from abc import ABC

from domain.base_document import DocumentManager, Document
from numpy import array
import twitter
from datetime import datetime, timezone
import dateutil.parser
import dateutil.tz


class TweetManager(DocumentManager):

    def __init__(self):
        super().__init__(directory='twitter', filename='tweets.csv', encoding='utf-8', headers=['Id', 'Author', 'Date', 'Text'])
        # initialize api instance
        self.twitter_api = twitter.Api(consumer_key='***REMOVED***',
                                       consumer_secret='***REMOVED***',
                                       access_token_key='***REMOVED***',
                                       access_token_secret='***REMOVED***',
                                       sleep_on_rate_limit=True)

        # test authentication
        print(self.twitter_api.VerifyCredentials())

    def create_document(self, file_row):
        return Document(file_row[0], file_row[1], file_row[2], file_row[3], None)

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

                [tweets_fetched.append(self.create_document([x.id, twitter_user, self.reformat_date(x.created_at), x.text, None]))
                 for x in response if x.lang == 'en' and dateutil.parser.parse(x.created_at) > datetime(2019, 12, 1, 0, 0, 0, tzinfo=timezone.utc)]
                if len(response) == 0:
                    break
                max_id = response[-1].id
            except Exception as e:
                    pass
        return tweets_fetched

    def get_more_tweets(self, authors):
        new_tweets = []
        existing_tweets = self.load_documents()  # load stored tweets
        for author in authors:
            since_id = None
            if len(existing_tweets) > 0:
                author_tweets = list(filter(lambda t: t.author == author, existing_tweets))
                if len(author_tweets) > 0:
                    latest_tweet = max(author_tweets, key=lambda x: x.date)
                    since_id = latest_tweet.id
            new_tweets.extend(self.api_get_user_timeline(author, count=1000, since_id=since_id))
        return new_tweets


class TrainingDataManager(DocumentManager):

    def __init__(self):
        super().__init__(directory='twitter', filename='training.csv', encoding='latin-1')

    def create_document(self, file_row):
        # mapping ['Target', 'Id', 'Date', 'flag', 'Author', 'Text'] to namedtuple('Document', ['id', 'author', 'date', 'text', 'label'])
        return Document(file_row[1], file_row[4], file_row[2], file_row[5], file_row[0])