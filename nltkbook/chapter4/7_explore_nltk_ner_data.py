from nltk.corpus import conll2002

# Language-independent named entity recognition
print(conll2002.chunked_sents()[0])

from nltk.corpus import ieer

# XML documents without POS tags
print(ieer.raw('APW_19980424'))


