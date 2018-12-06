from nltk.tokenize.casual import TweetTokenizer


tweet = "@BeaMiller u didn't follow me :(("

tokenizer = TweetTokenizer()
print(tokenizer.tokenize(tweet))
# ['@BeaMiller', 'u', "didn't", 'follow', 'me', ':(', '(']

tokenizer = TweetTokenizer(strip_handles=True)
print(tokenizer.tokenize(tweet))
# ['u', "didn't", 'follow', 'me', ':(', '(']