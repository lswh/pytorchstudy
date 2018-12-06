from sklearn.svm import LinearSVC
from classify import ScikitClassifier


def train_scikit_classifier(dataset):
    """
    dataset = list of tuples: [({feature1: value1, ...}, label), ...]
    """
    # split the dataset into featuresets and the predicted labels
    featuresets, labels = zip(*dataset)

    classifier = ScikitClassifier(classifier=LinearSVC())
    classifier.train(featuresets, labels)
    return classifier


import time
from nltk.tag import ClassifierBasedTagger
from utils import read_ud_pos_data
from tag import pos_features


if __name__ == "__main__":
    print("Loading data ...")
    train_data = list(read_ud_pos_data('../../../data/en-ud-train.conllu'))
    test_data = list(read_ud_pos_data('../../../data/en-ud-dev.conllu'))
    print("train_data", train_data)
    print("Data loaded .")

    start_time = time.time()
    print("Starting training ...")
    tagger = ClassifierBasedTagger(
        feature_detector=pos_features,
        train=train_data[:2000],
        classifier_builder=train_scikit_classifier,
    )
    end_time = time.time()
    print("Training complete. Time={0:.2f}s".format(end_time - start_time))


    print("Computing test set accuracy ...")
    print(tagger.evaluate(test_data))  # 0.8949021790997296

    print(tagger.tag("This is a test".split()))