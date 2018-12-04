from nltk import bigrams, trigrams, word_tokenize text = "John works at Intel."
tokens = word_tokenize(text)
print(list(bigrams(tokens))) # the `bigrams` function returns a generator, so we must unwind it
# [('John', 'works'), ('works', 'at'), ('at', 'Intel'), ('Intel', '.')]
print(list(trigrams(tokens))) # the `trigrams` function returns a generator, so we must unwind it # [('John', 'works', 'at'), ('works', 'at', 'Intel'), ('at', 'Intel', '.')]
