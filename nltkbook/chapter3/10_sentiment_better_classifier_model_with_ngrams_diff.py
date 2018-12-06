# ...

vectorizer = CountVectorizer(lowercase=True,
                             tokenizer=tweet_tokenizer.tokenize,
                             ngram_range=(1, 3))

# ...