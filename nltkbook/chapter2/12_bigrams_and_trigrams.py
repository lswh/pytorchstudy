import nltk
import collections
from pprint import pprint

text = """
How much wood does a woodchuck chuck
if a woodchuck could chuck wood
"""

bigram_features = collections.Counter(
    list(nltk.bigrams(text.lower().split())))

trigram_features = collections.Counter(
    list(nltk.trigrams(text.lower().split())))


pprint(bigram_features)
pprint(trigram_features)

{
    ('a', 'woodchuck'): 2,
    ('how', 'much'): 1,
    ('much', 'wood'): 1,
    ('wood', 'does'): 1,
    ('does', 'a'): 1,
    ('woodchuck', 'chuck'): 1,
    ('chuck', 'if'): 1,
    ('if', 'a'): 1,
    ('woodchuck', 'could'): 1,
    ('could', 'chuck'): 1,
    ('chuck', 'wood'): 1
}

{
    ('how', 'much', 'wood'): 1,
    ('much', 'wood', 'does'): 1,
    ('wood', 'does', 'a'): 1,
    ('does', 'a', 'woodchuck'): 1,
    ('a', 'woodchuck', 'chuck'): 1,
    ('woodchuck', 'chuck', 'if'): 1,
    ('chuck', 'if', 'a'): 1,
    ('if', 'a', 'woodchuck'): 1,
    ('a', 'woodchuck', 'could'): 1,
    ('woodchuck', 'could', 'chuck'): 1,
    ('could', 'chuck', 'wood'): 1
}
