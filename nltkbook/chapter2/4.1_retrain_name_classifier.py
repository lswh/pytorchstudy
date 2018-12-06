import nltk
import random
from nltk.corpus import names


def extract_features(name):
    """
    Get the features used for name classification
    """
    return {
        'last_letter': name[-1],
        'vowel_count': len([c for c in name if c in 'AEIOUaeiou'])
    }


# Get the names
boy_names = names.words('male.txt')
girl_names = names.words('female.txt')

# Build the dataset
boy_names_dataset = [(extract_features(name), 'boy') for name in boy_names]
girl_names_dataset = [(extract_features(name), 'girl') for name in girl_names]

# Put all the names together
data = boy_names_dataset + girl_names_dataset

# Mix everything together
random.shuffle(data)

# Let's take a look at the first few entries
print(data[:10])

# Split the dataset into training data and test data
train_data, test_data = data[:int(0.75 * len(data))], data[int(0.75 * len(data)) + 1:]

name_classifier = nltk.DecisionTreeClassifier.train(train_data)

# Let's take if for a spin
print(name_classifier.classify(extract_features('Bono')))  # boy
print(name_classifier.classify(extract_features('Latiffa')))  # girl
print(name_classifier.classify(extract_features('Gaga')))  # girl
print(name_classifier.classify(extract_features('Joey')))  # girl

# Sorry Joey :(
# Let's test how well it performs on the test data
print(nltk.classify.accuracy(name_classifier, test_data))  # 0.7420654911838791

# Let's have a look at the tree
print(name_classifier.pretty_format())