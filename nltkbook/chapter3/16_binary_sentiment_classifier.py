import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


data = pd.read_csv('./twitter_sentiment_analysis.csv')
# filter out the `neutrals`
data = data[(data['sentiment'].isin(['positive', 'negative']))]

tweets = data['tweet'].values.astype(str)
sentiments = data['sentiment'].values.astype(str)

# Split the data for training and for testing and shuffle it
X_train, X_test, y_train, y_test = train_test_split(tweets, sentiments,
                                                    test_size=0.2, shuffle=True)

vectorizer = CountVectorizer(lowercase=True)
vectorizer.fit(X_train)

X_train_vectorized = vectorizer.transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

classifier = MultinomialNB()
classifier.fit(X_train_vectorized, y_train)

# Compute our classifier's accuracy
print(accuracy_score(y_test, classifier.predict(X_test_vectorized)))


from sklearn.metrics import recall_score, precision_score, f1_score


print(recall_score(
    y_test, classifier.predict(X_test_vectorized), pos_label='negative'))
# 0.945172564439

print(precision_score(
    y_test, classifier.predict(X_test_vectorized), pos_label='negative'))
# 0.828768435166

print(f1_score(
    y_test, classifier.predict(X_test_vectorized), pos_label='negative'))
# 0.883151341974


from sklearn.metrics import classification_report

print(classification_report(y_test, classifier.predict(X_test_vectorized)))

#              precision    recall  f1-score   support
#
#    negative       0.82      0.94      0.88      4499
#    positive       0.80      0.54      0.64      1955
#
# avg / total       0.82      0.82      0.81      6454