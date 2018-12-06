import pandas as pd
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize.casual import TweetTokenizer


tweet_tokenizer = TweetTokenizer(strip_handles=True)

data = pd.read_csv('./twitter_sentiment_analysis.csv')

tweets = data['tweet'].values.astype(str)
sentiments = data['sentiment'].values.astype(str)

# Shuffle the data
tweets, sentiments = shuffle(tweets, sentiments)

# Put everything in a Pipeline
pipeline = Pipeline([
    ('vectorizer', CountVectorizer(
        lowercase=True,
        tokenizer=tweet_tokenizer.tokenize,
        ngram_range=(1, 3))),
    ('classifier', LogisticRegression())
])

classifier = GridSearchCV(pipeline, {
    # try out different ngram ranges
    'vectorizer__ngram_range': ((1, 2), (2, 3), (1, 3)),
    # check if setting all non zero counts to 1 makes a difference
    'vectorizer__binary': (True, False),
}, n_jobs=-1, verbose=True, error_score=0.0, cv=5)

# Compute the vocabulary and train the classifier
classifier.fit(tweets, sentiments)

print("Best Accuracy: ", classifier.best_score_)
print("Best Parameters: ", classifier.best_params_)

# Best Accuracy:  0.81920859947
# Best Parameters:  {'vectorizer__binary': True, 'vectorizer__ngram_range': (1, 3)}

from nltk.stem import PorterStemmer
from nltk.sentiment.util import mark_negation

stemmer = PorterStemmer()


def stemming_tokenizer(text):
    return [stemmer.stem(t) for t in tweet_tokenizer.tokenize(text)]


def tokenizer_negation_aware(text):
    return mark_negation(tweet_tokenizer.tokenize(text))


def stemming_tokenizer_negation_aware(text):
    return mark_negation(stemming_tokenizer(text))

tweet = "@rebeccalowrie No, it's not just you.  Twitter have decided to take that feature away :-("

print(tweet_tokenizer.tokenize(tweet))
# ['No', ',', "it's", 'not', 'just', 'you', '.', 'Twitter', 
# 'have', 'decided', 'to', 'take', 'that', 'feature', 'away', ':-(']

print(stemming_tokenizer(tweet))
# ['No', ',', "it'", 'not', 'just', 'you', '.', 'twitter', 
# 'have', 'decid', 'to', 'take', 'that', 'featur', 'away', ':-(']

print(tokenizer_negation_aware(tweet))
# ['No', ',', "it's", 'not', 'just_NEG', 'you_NEG', '.', 'Twitter', 
# 'have', 'decided', 'to', 'take', 'that', 'feature', 'away', ':-(']

print(stemming_tokenizer_negation_aware(tweet))
# ['No', ',', "it'", 'not', 'just_NEG', 'you_NEG', '.', 'twitter', 
# 'have', 'decid', 'to', 'take', 'that', 'featur', 'away', ':-(']

classifier = GridSearchCV(pipeline, {
    'vectorizer__tokenizer': (
        tweet_tokenizer.tokenize,
        stemming_tokenizer,
        tokenizer_negation_aware,
        stemming_tokenizer_negation_aware,
    )
}, n_jobs=-1, verbose=True, error_score=0.0, cv=5)

# Compute the vocabulary and train the classifier
classifier.fit(tweets, sentiments)

print("Best Accuracy: ", classifier.best_score_)
print("Best Parameters: ", classifier.best_params_)

# Best Accuracy:  0.820688580776
# Best Parameters:  {'vectorizer__tokenizer': <function stemming_tokenizer at 0x10c519488>}

from sklearn.externals import joblib
clf = classifier.best_estimator_
joblib.dump(clf, './twitter_sentiment.joblib')
