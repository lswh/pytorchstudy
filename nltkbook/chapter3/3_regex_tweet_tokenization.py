import re
import nltk


tweet = "@BeaMiller u didn't follow me :(("

print(re.findall(r"(?u)\b\w\w+\b", tweet))
# ['BeaMiller', 'didn', 'follow', 'me']

print(nltk.word_tokenize(tweet))
# ['@', 'BeaMiller', 'u', 'did', "n't", 'follow', 'me', ':', '(', '(']