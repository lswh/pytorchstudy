import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


data = pd.read_csv('./twitter_sentiment_analysis.csv')
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
# 0.698818642087

from sklearn.metrics import recall_score, precision_score, f1_score


print(recall_score(
    y_test, classifier.predict(X_test_vectorized), average='weighted'))
# 0.698818642087

print(precision_score(
    y_test, classifier.predict(X_test_vectorized), average='weighted'))
# 0.687808952468

print(f1_score(
    y_test, classifier.predict(X_test_vectorized), average='weighted'))
# 0.670033645784


from sklearn.metrics import classification_report

print(classification_report(
    y_test, classifier.predict(X_test_vectorized)))

#              precision    recall  f1-score   support
#
#    negative       0.71      0.92      0.80      4530
#     neutral       0.51      0.25      0.33      1238
#    positive       0.76      0.48      0.59      1935
#
# avg / total       0.69      0.70      0.67      7703



from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_test, classifier.predict(X_test_vectorized),
                      labels=['positive', 'neutral', 'negative'])
plt.show(block=True)

sns.heatmap(cm, square=True, annot=True, cbar=False, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show(block=True)

# normalize the confusion matrix
cm = cm.astype(float) / cm.sum(axis=0)

print(cm)
sns.heatmap(cm, square=True, annot=True, cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show(block=True)
