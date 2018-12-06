# ...
from nltk.tokenize.casual import TweetTokenizer


tweet_tokenizer = TweetTokenizer(strip_handles=True)

# ...

vectorizer = CountVectorizer(lowercase=True, tokenizer=tweet_tokenizer.tokenize)

# ...