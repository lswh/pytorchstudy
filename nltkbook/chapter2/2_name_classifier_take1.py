import nltk
import random
from nltk.corpus import names


def extract_features(name):
    """
    Get the features used for name classification
    """
    return {
        'last_letter': name[-1]
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
cutoff = int(0.75 * len(data))
train_data, test_data = data[:cutoff], data[cutoff + 1:]

# Let's train probably the most popular classifier in the world
name_classifier = nltk.DecisionTreeClassifier.train(train_data)

# Take if for a spin
print(name_classifier.classify(extract_features('Bono')))  # boy
print(name_classifier.classify(extract_features('Latiffa')))  # girl
print(name_classifier.classify(extract_features('Gaga')))  # girl
print(name_classifier.classify(extract_features('Joey')))  # girl

# Sorry Joey :(
# Test how well it performs on the test data: 
# Accuracy = correctly_labelled_samples / all_samples
print(nltk.classify.accuracy(name_classifier, test_data))  # 0.7420654911838791

# Look at the tree
print(name_classifier.pretty_format())