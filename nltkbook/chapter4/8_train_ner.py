import random
import itertools
from nltk.corpus import conll2000
from nltk import word_tokenize, pos_tag
from sklearn.linear_model import Perceptron
from sklearn.feature_extraction import FeatureHasher
from chunk import ClassifierBasedChunkParser
from classify import ScikitClassifier
from utils import read_gmb_ner
from ner import ner_features


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

    batch = list(itertools.islice(sentences, batch_size))
    dataset = feature_detector(batch)

    # split the dataset into featuresets and the predicted labels
    featuresets, labels = zip(*dataset)

    # This vectorizer doesn't need to be fitted
    vectorizer = FeatureHasher(n_features=1000000)

    classifier = Perceptron(tol=0.00001, max_iter=25, n_jobs=-1)

    for _ in range(max_iterations):
        current_corpus_iterator, sentences = itertools.tee(sentences)
        batch_count = 0

        while True:
            batch_count += 1
            print("Training on batch={0}".format(batch_count))
            classifier.partial_fit(vectorizer.transform(featuresets), labels, ALL_LABELS)

            batch = list(itertools.islice(current_corpus_iterator, batch_size))
            if not batch:
                break

            dataset = feature_detector(batch)
            featuresets, labels = zip(*dataset)

    scikit_classifier = ScikitClassifier(classifier=classifier, vectorizer=vectorizer)

    return scikit_classifier


if __name__ == "__main__":
    # Prepare the training and the test set
    conll_sents = list(conll2000.chunked_sents())
    random.shuffle(conll_sents)
    train_sents = conll_sents[:int(len(conll_sents) * 0.9)]
    test_sents = conll_sents[int(len(conll_sents) * 0.9 + 1):]

    train_sents = read_gmb_ner('../../../data/gmb-2.2.0/data', start_index=0, end_index=8999)
    test_sents = read_gmb_ner('../../../data/gmb-2.2.0/data', start_index=9000, end_index=9999)

    print("Training Classifier")
    classifier_chunker = ClassifierBasedChunkParser(
        train_sents,
        ner_features,
        lambda iterator, detector: incremental_train_scikit_classifier(iterator, detector, 500, 2),
    )
    print("Classifier Trained")
    print(classifier_chunker.evaluate(test_sents))

    print(classifier_chunker.parse(
        pos_tag(word_tokenize(
            "Twitter Inc. is based in San Francisco , California , United States , "
            "and has more than 25 offices around the world ."))))

    from nltk import tree2conlltags

    chunked = tree2conlltags(classifier_chunker.parse(
        pos_tag(word_tokenize(
            "Twitter Inc. is based in San Francisco , California , United States , "
            "and has more than 25 offices around the world ."))))

    print(chunked)

    classifier_chunker.parse(
        pos_tag(word_tokenize(
            "Twitter Inc. is based in San Francisco , California , United States , "
            "and has more than 25 offices around the world ."))).draw()

    print(classifier_chunker.tagger.classifier().labels())