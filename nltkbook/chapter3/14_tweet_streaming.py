import tweepy
import webbrowser
from collections import Counter
from sklearn.externals import joblib


class SentimentAnalysisStreamListener(tweepy.StreamListener):
    def __init__(self, model_path):
        self.counts = Counter()
        self.sentiment_classifier = joblib.load(model_path)
        super(SentimentAnalysisStreamListener, self).__init__()

    # this method will be called when a tweet we are interested in is published
    def on_status(self, status):
        sentiment = self.sentiment_classifier.predict([status.text])[0]
        self.counts[sentiment] += 1
        tweet_count = sum(self.counts.values())

        if not tweet_count % 100:
            print({k: v/tweet_count for k, v in self.counts.items()})
            print("Tweets Collected: %d" % tweet_count)


# Depending on how we saved the model we might need to redefine this
from nltk import PorterStemmer, TweetTokenizer


stemmer = PorterStemmer()
tweet_tokenizer = TweetTokenizer(strip_handles=True)


def stemming_tokenizer(text):
    return [stemmer.stem(t) for t in tweet_tokenizer.tokenize(text)]


if __name__ == "__main__":
    auth = tweepy.OAuthHandler('YOUR_TOKEN', 'YOUR_SECRET')

    redirect_url = auth.get_authorization_url()
    webbrowser.open_new(redirect_url)

    verifier = input('Verifier:')
    auth.get_access_token(verifier)

    listener = SentimentAnalysisStreamListener(
        model_path='./twitter_sentiment.joblib')
    # this registers our Listener as a handler for tweets
    stream = tweepy.Stream(auth=auth, listener=listener)

    stream.filter(track=['trump']) # I've selected "trump" here as a filter for tweets