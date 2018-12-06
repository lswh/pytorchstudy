from sklearn.feature_extraction.text import CountVectorizer


text = """
How much wood does a woodchuck chuck
if a woodchuck could chuck wood
"""

vectorizer = CountVectorizer(lowercase=True)

# "train" the vectorizer, aka compute the vocabulary
vectorizer.fit([text])

# transform text to features
print(vectorizer.transform([text]))

#  (0, 0)	2
#  (0, 1)	1
#  (0, 2)	1
#  (0, 3)	1
#  (0, 4)	1
#  (0, 5)	1
#  (0, 6)	2
#  (0, 7)	2

result = vectorizer.transform(["Unseen words", "BLT sandwich"])
print(type(result), result, result.shape)