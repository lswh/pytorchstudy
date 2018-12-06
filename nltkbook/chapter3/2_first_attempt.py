import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

data = pd.read_csv('./twitter_sentiment_analysis.csv')

tweets = data['tweet'].values.astype(str)
sentiments = data['sentiment'].values.astype(str)

# Split the data for training and for testing and shuffle it
X_train, X_test, y_train, y_test = train_test_split(tweets, sentiments,
                                                    test_size=0.2, shuffle=True)

vectorizer = CountVectorizer(lowercase=True)

# Compute the vocabulary only on the training data
vectorizer.fit(X_train)

# Transform the text list to a matrix form
X_train_vectorized = vectorizer.transform(X_train)

classifier = MultinomialNB()

# Train the classifier
classifier.fit(X_train_vectorized, y_train)

# Vectorize the test data
X_test_vectorized = vectorizer.transform(X_test)

# Check our classifier performance
score = classifier.score(X_test_vectorized, y_test)
print("Accuracy=", score)
