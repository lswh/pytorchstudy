import nltk
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder 
from nltk.collocations import TrigramAssocMeasures, TrigramCollocationFinder


bigram_measures = BigramAssocMeasures() 
trigram_measures = TrigramAssocMeasures()


# Compute length-2 collocations
finder = BigramCollocationFinder.from_words(nltk.corpus.reuters.words()) # only bigrams that appear 5+ times
finder.apply_freq_filter(5)

# return the 50 bigrams with the highest PMI (Pointwise Mutual Information)
print(finder.nbest(bigram_measures.pmi, 50))

# among the collocations we can find stuff like: (u'Corpus', u'Christi') ...

# Compute length-3 collocations
finder = nltk.collocations.TrigramCollocationFinder.from_words(nltk.corpus.reuters.words()) # only trigrams that appear 5+ times

finder.apply_freq_filter(5)
# return the 50 trigrams with the highest PMI
print(finder.nbest(trigram_measures.pmi, 50))
# among the collocations we can find stuff like: (u'Special', u'Drawing', u'Rights')
