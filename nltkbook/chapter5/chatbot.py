import nltk
import random
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from manuscript.snippets.chapter4.chunk import ClassifierBasedChunkParser
from manuscript.snippets.chapter4.chunk import chunk_features
from manuscript.snippets.chapter4.classify import ScikitClassifier
from nltk.tokenize.casual import TweetTokenizer

from datasets import MOVIE_EXPERT_TRAINING_SET


tokenizer = TweetTokenizer()


def train_intent_model(train_set):
    pipeline = Pipeline(steps=[
        ('vectorizer', CountVectorizer(tokenizer=tokenizer.tokenize)),
        ('classifier', MultinomialNB()),
    ])

    dataset = []
    for intent, samples in train_set.items():
        for sample in samples:
            dataset.append((sample[0], intent))

    random.shuffle(dataset)
    intent_X, intent_y = zip(*dataset)

    pipeline.fit(intent_X, intent_y)
    print("Accuracy on training set:", pipeline.score(intent_X, intent_y))
    return pipeline


train_intent_model(MOVIE_EXPERT_TRAINING_SET)


def sub_list(lst, slst):
    """
    Utility function for getting
    the index of a sublist inside a list
    """

    start_index = 0
    while len(lst) > len(slst):
        if lst[:len(slst)] == slst:
            return start_index

        lst, start_index = lst[1:], start_index + 1

    return None


def mark_entities(tagged_sentence, entity_words, label):
    """
    tagged_sentence: [('Word', 'Tag'), ...]
    entity_words: ['This', 'is', 'an', 'entity']
    label: the entity type

    return a nltk.Tree instance with the entities wrapped in chunks
    """

    iob_tagged = [(w, t, 'O') for w, t in tagged_sentence]

    words = nltk.untag(tagged_sentence)
    start_index = sub_list(words, entity_words)
    if start_index is not None:
        iob_tagged[start_index] = (
            iob_tagged[start_index][0],
            iob_tagged[start_index][1],
            'B-' + label
        )
        for idx in range(1, len(entity_words)):
            iob_tagged[start_index + idx] = (
                iob_tagged[start_index + idx][0],
                iob_tagged[start_index + idx][1],
                'I-' + label
            )

    return nltk.conlltags2tree(iob_tagged)


def build_extractors_datasets(train_set):
    """
    Transform the training set from the original form nltk.Tree organized by entity_type
    {entity_type: [tree1, tree2, ...]}
    """
    tokenizer = TweetTokenizer()
    datasets = defaultdict(list)
    for _, samples in train_set.items():
        for sample in samples:
            words = tokenizer.tokenize(sample[0])
            tagged = nltk.pos_tag(words)

            for entity_type, entity in sample[1].items():
                entity_words = tokenizer.tokenize(entity)

                tree = mark_entities(tagged, entity_words, entity_type)
                datasets[entity_type].append(tree)

    return datasets


def build_intent_extractor_mapping(train_set):
    mapping = {}
    for intent in train_set:
        mapping[intent] = set({})
        for sample in train_set[intent]:
            mapping[intent].update(list(sample[1].keys()))

    return mapping


def train_extractor(trees):
    def train_scikit_classifier(sentences, feature_detector):

        dataset = feature_detector(sentences)
        featuresets, labels = zip(*dataset)

        scikit_classifier = ScikitClassifier()
        scikit_classifier.train(featuresets, labels)
        return scikit_classifier

    classifier_chunker = ClassifierBasedChunkParser(
        trees,
        chunk_features,
        train_scikit_classifier,
    )

    print("Accuracy on training set:", classifier_chunker.evaluate(trees))
    return classifier_chunker

extractor_datasets = build_extractors_datasets(MOVIE_EXPERT_TRAINING_SET)
movie_extractor = train_extractor(extractor_datasets['movie'])
actor_extractor = train_extractor(extractor_datasets['actor'])

print(movie_extractor.parse(nltk.pos_tag(tokenizer.tokenize("I want to see a movie like Indiana Jones!"))))
print(movie_extractor.parse(nltk.pos_tag(tokenizer.tokenize("Who is the director of Ninja Turtles!"))))
print(movie_extractor.parse(nltk.pos_tag(tokenizer.tokenize("When was Indiana Jones released?"))))
print(actor_extractor.parse(nltk.pos_tag(tokenizer.tokenize("In what movies did Selena Gomez act?"))))


class Chatbot(object):
    def __init__(self, min_intent_confidence=.2):
        self.min_intent_confidence = min_intent_confidence

        self.intent_model = None
        self.entity_types = []
        self.extractors = {}
        self.intent_extractor_mapping = None
        self.handlers = {}

    def train(self, training_set):
        # Train intent model
        self.intent_model = train_intent_model(training_set)

        extractor_datasets = build_extractors_datasets(training_set)

        self.intent_extractor_mapping = build_intent_extractor_mapping(training_set)

        # Train extractor for each entity type
        for entity_type, data in extractor_datasets.items():
            self.extractors[entity_type] = train_extractor(data)

    def predict_intent(self, line):
        probs = self.intent_model.predict_proba([line])[0]
        best_score_index = probs.argmax()
        best_score = probs[best_score_index]

        if best_score < self.min_intent_confidence:
            return None

        return self.intent_model.classes_[best_score_index]

    def predict_entities(self, line, intent):
        tagged = nltk.pos_tag(tokenizer.tokenize(line))

        # Get applicable extractors for the intent
        applicable_extractors = self.intent_extractor_mapping[intent]

        # Apply the extractors and get the entities
        entities = {}

        for extractor_name in applicable_extractors:
            tree = self.extractors[extractor_name].parse(tagged)
            extracted_entities = [t for t in tree if isinstance(t, nltk.Tree)]

            if extracted_entities:
                first_entity = extracted_entities[0]
                entities[extractor_name] = ' '.join(t[0] for t in first_entity)

        return entities

    def handle(self, intent, entities):
        assert intent in self.handlers, "Register a handler for intent: %s" % intent
        return self.handlers[intent](entities)

    def register_handler(self, intent):
        """ Decorator for registering an intent handler """
        def wrapper(handler):
            self.handlers[intent] = handler
        return wrapper

    def register_default_handler(self):
        """ Decorator for registering the default (fallback) handler """
        def wrapper(handler):
            self.handlers[None] = handler
        return wrapper

    def tell(self, line):
        intent = self.predict_intent(line)

        # In case we haven't registered a handler for this intent
        # let's just pretend we just don't know better
        if intent not in self.handlers:
            intent = None

        if intent is not None:
            entities = self.predict_entities(line, intent)
        else:
            entities = {}

        return self.handle(intent, entities)