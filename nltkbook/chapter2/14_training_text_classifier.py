import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Let's print a snippet of the documentation for Naive Bayes classifier
print(MultinomialNB.__doc__[:415])

# Remember this is where we saved all the data we crawled previously
data = pd.read_csv('./text_analysis_data.csv')

# Where we keep the actual texts
text_samples, labels = []
for idx, row in data.iterrows():
    with open('./clean_data/{0}'.format(row['file_name']), 'r') as text_file:
        text = text_file.read()
        text_samples.append(text)
        labels.append(row['category'])

# Split the data for training and for testing and shuffle it
# keep 20% for testing, and use the rest for training
# shuffling is important because the classes are not random in our dataset
labels = data['category'].as_matrix()
X_train, X_test, y_train, y_test = train_test_split(
    text_samples, labels, test_size=0.2, shuffle=True)

vectorizer = CountVectorizer(lowercase=True)

# Compute the vocabulary using only the training data
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

# Pick a text from the test samples
import random
random_choice = random.randint(0, len(X_test))
text, label = X_test[random_choice], y_test[random_choice]

# Check the text we picked and the expected label
print(text, label)

# Let's use the classifier to predict a label
text_vectorized = vectorizer.transform([text])

# As other scikit-learn methods, predict works on matrices
print(classifier.predict(text_vectorized))

import time
from sklearn.externals import joblib

timestamp = int(time.time())
# Save the vectorizer
joblib.dump(vectorizer, './text_analysis_vectorizer_%s.joblib' % timestamp)

# Save the classifier
joblib.dump(classifier, './text_analysis_classifier_%s.joblib' % timestamp)

# Load the vectorizer
vectorizer = joblib.load('./text_analysis_vectorizer_%s.joblib' % timestamp)

# Load the classifier
classifier = joblib.load('./text_analysis_classifier_%s.joblib' % timestamp)

# Test the loaded component by classifying a text
print(classifier.predict(vectorizer.transform([
    "Marketing for Business Professionals"])))