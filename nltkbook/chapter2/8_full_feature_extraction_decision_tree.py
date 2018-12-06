import string
from nltk.corpus import names
from sklearn.model_selection import train_test_split


def extract_features(name):
    """
    Get the features used for name classification
    """
    features = {
        # Last letter
        'last_letter': name[-1],
        # First letter
        'first_letter': name[0],
        # How many vowels
        'vowel_count': len([c for c in name if c in 'AEIOUaeiou'])
    }
    # Build letter and letter count features
    for c in string.ascii_lowercase:
        features['contains_' + c] = c in name
        features['count_' + c] = name.lower().count(c)
    return features

# Get the names
boy_names = names.words('male.txt')
girl_names = names.words('female.txt')

# Build the dataset
boy_names_dataset = [(extract_features(name), 'boy') for name in boy_names]
girl_names_dataset = [(extract_features(name), 'girl') for name in girl_names]

# Put all the names together
data = boy_names_dataset + girl_names_dataset

# Split the data in features and classes
X, y = list(zip(*data))

# split and randomize
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, shuffle=True)
print(X_train)
print(y_train)

from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer

dict_vectorizer = DictVectorizer()
name_classifier = DecisionTreeClassifier()

# Scikit-Learn models work with arrays not dicts
# We need to train the vectorizer so that
# it knows what's the format of the dicts
dict_vectorizer.fit(X_train)

# Vectorize the training data
X_train_vectorized = dict_vectorizer.transform(X_train)

# Train the classifier on vectorized data
name_classifier.fit(X_train_vectorized, y_train)

# Test the model
X_test_vectorized = dict_vectorizer.transform(X_test)
print(name_classifier.score(X_test_vectorized, y_test))

# Let's take a look at how the transformed data looks like:
NAMES = ['Lara', 'Carla', 'Ioana', 'George', 'Steve', 'Stephan']
transformed = dict_vectorizer.transform([extract_features(name) for name in NAMES])

# We get a scipy sparse matrix, which is a bit hard to read
print(transformed)
print(type(transformed)) # <class 'scipy.sparse.csr.csr_matrix'>

# We can transform the data back
print(dict_vectorizer.inverse_transform(transformed))

# We can check the feature names
print(dict_vectorizer.feature_names_)
# ['contains_a', 'contains_b', 'contains_c', 'contains_d', 'contains_e', 'contains_f', ...

# Just as an excercise, let's build a feature dictionary in reverse:
import numpy as np

# build a vector with 0 value for each feature
vectorized = np.zeros(len(dict_vectorizer.feature_names_))

# Let's set the features by hand for an imagined name: "Wwbwi"
# The index in `feature_names_` represents the index of the feature
# in a row of the sparse matrix we previously discussed

# Our name has the last letter `i`
vectorized[dict_vectorizer.feature_names_.index('last_letter=i')] = 1.0

# Contains 3 `w`s 
vectorized[dict_vectorizer.feature_names_.index('count_w')] = 3.0
vectorized[dict_vectorizer.feature_names_.index('count_b')] = 1.0
vectorized[dict_vectorizer.feature_names_.index('count_i')] = 1.0

# It contains the letter `b`, `i`, `w`
vectorized[dict_vectorizer.feature_names_.index('contains_b')] = 1.0
vectorized[dict_vectorizer.feature_names_.index('contains_w')] = 1.0
vectorized[dict_vectorizer.feature_names_.index('contains_i')] = 1.0

print(vectorized)
# Let's see what we've built:
# [ 0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  3.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]

# Let's now apply the inverse transformation, to get the feature dict
print(dict_vectorizer.inverse_transform([vectorized]))
# [{'contains_a': 1.0, 'contains_b': 1.0, 'contains_w': 1.0, 
#   'count_a': 1.0, 'count_b': 1.0, 'count_w': 3.0, 'last_letter=a': 1.0}]

# Let's make some predictions
NAMES = ['Lara', 'Carla', 'Ioana', 'George', 'Steve', 'Stephan']
transformed = dict_vectorizer.transform([extract_features(name) for name in NAMES])
print(name_classifier.predict(transformed))
# We get: ['girl' 'girl' 'girl' 'girl' 'boy' 'boy']
# Oops, sorry George!




