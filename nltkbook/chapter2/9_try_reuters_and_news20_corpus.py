# Let's try out Reuters corpus
from nltk.corpus import reuters


# Let's see what are the Reuters categories
print(reuters.categories())
# ['acq', 'alum', 'barley', 'bop', 'carcass', 'castor-oil', 'cocoa', ...


# Let's check out the 20 newsgroups dataset
from sklearn.datasets import fetch_20newsgroups
news20 = fetch_20newsgroups(subset='train')
print(list(news20.target_names))
# ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', ...
