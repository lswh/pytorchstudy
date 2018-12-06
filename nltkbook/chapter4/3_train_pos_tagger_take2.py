import time
import itertools
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import Perceptron
from classify import ScikitClassifier
from classify import ClassifierBasedTaggerBatchTrained
from tag import pos_features
from utils import read_ud_pos_data


def incremental_train_scikit_classifier(
        sentences,
        feature_detector,
        batch_size,
        max_iterations):

    initial_corpus_iterator, sentences = itertools.tee(sentences)

    # compute all labels
    ALL_LABELS = set([])

    for sentence in initial_corpus_iterator:
        for w, t in sentence:
            ALL_LABELS.add(t)

    ALL_LABELS = list(ALL_LABELS)

    # This vectorizer doesn't need to be fitted
    vectorizer = FeatureHasher(n_features=1000000)

    classifier = Perceptron(tol=0.00001, max_iter=25, n_jobs=-1)

    for _ in range(max_iterations):
        current_corpus_iterator, sentences = itertools.tee(sentences)
        batch_count = 0

        while True:
            batch = list(itertools.islice(current_corpus_iterator, batch_size))
            if not batch:
                break
            batch_count += 1
            print("Training on batch={0}".format(batch_count))

            dataset = feature_detector(batch)

            # split the dataset into featuresets and the predicted labels
            featuresets, labels = zip(*dataset)

            classifier.partial_fit(vectorizer.transform(featuresets), labels, ALL_LABELS)

    scikit_classifier = ScikitClassifier(classifier=classifier, vectorizer=vectorizer)

    return scikit_classifier


if __name__ == "__main__":
    test_data = read_ud_pos_data('../../../data/en-ud-dev.conllu')

    start_time = time.time()
    print("Starting training ...")

    tagger = ClassifierBasedTaggerBatchTrained(
        feature_detector=pos_features,
        train=read_ud_pos_data('../../../data/en-ud-train.conllu'),
        classifier_builder=lambda iterator, detector: incremental_train_scikit_classifier(
            iterator, detector, batch_size=500, max_iterations=30),
    )
    end_time = time.time()
    print("Training complete. Time={0:.2f}s".format(end_time - start_time))

    print("Computing test set accuracy ...")
    print(tagger.evaluate(test_data))  # 0.9255606807698425

    print(tagger.tag("This is a test".split()))