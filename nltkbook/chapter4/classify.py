import nltk
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


class ScikitClassifier(nltk.ClassifierI):
    """
    Wrapper over a scikit-learn classifier
    """
    def __init__(self, classifier=None, vectorizer=None, model=None):
        if model is None:
            if vectorizer is None:
                vectorizer = DictVectorizer(sparse=False)

            if classifier is None:
                classifier = LogisticRegression()

            self.model = Pipeline([
                ('vectorizer', vectorizer),
                ('classifier', classifier)
            ])
        else:
            self.model = model
            
    @property
    def vectorizer(self):
        return self.model[0][1]
    
    @property
    def classifier(self):
        return self.model[1][1]
    
    def train(self, featuresets, labels):
        self.model.fit(featuresets, labels)
        
    def partial_train(self, featuresets, labels, all_labels):
        self.model.partial_fit(featuresets, labels, all_labels)
        
    def test(self, featuresets, labels):
        self.model.score(featuresets, labels)
    
    def labels(self):
        return list(self.model.steps[1][1].classes_)
    
    def classify(self, featureset):
        return self.model.predict([featureset])[0]
    
    def classify_many(self, featuresets):
        return self.model.predict(featuresets)


from nltk import ClassifierBasedTagger
from nltk.metrics import accuracy


class ClassifierBasedTaggerBatchTrained(ClassifierBasedTagger):
    def _todataset(self, tagged_sentences):
        classifier_corpus = []
        for sentence in tagged_sentences:
            history = []
            untagged_sentence, tags = zip(*sentence)
            for index in range(len(sentence)):
                featureset = self.feature_detector(untagged_sentence,
                                                   index, history)
                classifier_corpus.append((featureset, tags[index]))
                history.append(tags[index])
        return classifier_corpus

    def _train(self, tagged_corpus, classifier_builder, verbose):
        """
        Build a new classifier, based on the given training data
        *tagged_corpus*.
        """

        if verbose:
            print('Constructing training corpus for classifier.')

        self._classifier = classifier_builder(tagged_corpus, lambda sents: self._todataset(sents))

    def evaluate(self, gold):
        dataset = self._todataset(gold)
        featuresets, tags = zip(*dataset)
        predicted_tags = self.classifier().classify_many(featuresets)
        return accuracy(tags, predicted_tags)
