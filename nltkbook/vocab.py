import nltk
fdist = nltk.FreqDist(nltk.corpus.reuters.words())
# top 10 most frequent words
print(fdist.most_common(n=10))
# [('.', 94687), (',', 72360), ('the', 58251), ('of', 35979), ('to', 34035), ('in', 26478), ('said', 25224), ('and', 2504\ 3), ('a', 23492), ('mln', 18037)]
# get the count of the word `stock` print(fdist['stock']) # 2346
# get the count of the word `stork` print(fdist['stork']) # 0 :(
# get the frequency of the word `the` print(fdist.freq('the')) # 0.033849129031826936
# get the words that only appear once (these words are called hapaxes)
print(fdist.hapaxes())
# Hapaxes usually are `mispeled` or weirdly `cApiTALIZED` words.
# Total number of distinct words print(len(fdist.keys())) # 41600
# Total number of samples print(fdist.N()) # 1720901
