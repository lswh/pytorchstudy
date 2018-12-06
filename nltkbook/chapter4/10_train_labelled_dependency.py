import random
import itertools
import numpy as np
from nltk import pos_tag, word_tokenize, ParserI
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier, Perceptron

from dependency import iter_ud_dependencies, dependency_features, DependencyParse, edge_label_features


class Transitions:
        SHIFT = 0
        LEFT_ARC = 1
        RIGHT_ARC = 2

        ALL = (SHIFT, LEFT_ARC, RIGHT_ARC)


DEPENDENCY_LABELS = [
    "acl",
    "acl:relcl",
    "advcl",
    "advmod",
    "amod",
    "appos",
    "aux",
    "aux:pass",
    "case",
    "cc",
    "cc:preconj",
    "ccomp",
    "compound",
    "compound:prt",
    "conj",
    "cop",
    "csubj",
    "csubj:pass",
    "dep",
    "det",
    "det:predet",
    "discourse",
    "dislocated",
    "expl",
    "fixed",
    "flat",
    "flat:foreign",
    "goeswith",
    "iobj",
    "list",
    "mark",
    "nmod",
    "nmod:npmod",
    "nmod:poss",
    "nmod:tmod",
    "nsubj",
    "nsubj:pass",
    "nummod",
    "obj",
    "obl",
    "obl:npmod",
    "obl:tmod",
    "orphan",
    "parataxis",
    "punct",
    "reparandum",
    "root",
    "vocative",
    "xcomp"
]


class ParserState(object):

    def __init__(self, parse):
        self.parse = parse

        # Put the first word of the parse in the stack
        self.stack = [1]

        # The buffer index is pointing to the second word
        self.buffer_index = 2

    def get_nodes(self, transition):
        if transition == Transitions.SHIFT:
            return None, None

        if transition == Transitions.RIGHT_ARC:
            return self.stack[-2], self.stack[-1]

        return self.buffer_index, self.stack[-1]

    def apply(self, transition, label=None):
        if transition == Transitions.SHIFT:
            self.stack.append(self.buffer_index)
            self.buffer_index += 1

            # No new edge was created
            return None, None

        # Add new edge
        if transition == Transitions.RIGHT_ARC:
            # New edge from `stack[-2]` to `stack[-1]`
            head, child = self.stack[-2], self.stack.pop()

        # transition == Transitions.LEFT_ARC
        else:
            # New edge from `buffer_index` to `stack[-1]`
            head, child = self.buffer_index, self.stack.pop()

        self.parse.add_edge(head, child, label)
        return head, child

    def next_valid(self):
        """ The list of transitions that are allowed in the current state """
        valid = []

        # We can do a SHIFT if the buffer is not empty
        if (self.buffer_index + 1) < len(self.parse):
            valid.append(Transitions.SHIFT)

        # We can do a RIGHT_ARC if there are at least 2 words in the stack
        if len(self.stack) >= 2:
            valid.append(Transitions.RIGHT_ARC)

        # We can do a LEFT_ARC if there is at least one node in the stack
        if len(self.stack) >= 1:
            valid.append(Transitions.LEFT_ARC)

        return valid

    def next_gold(self, gold_parse):
        """ What transitions would you choose in this state if you knew the final result """
        gold_heads = gold_parse.heads()
        buffer_nodes = range(self.buffer_index + 1, len(self.parse) - 1)

        # Let's start from the set of valid transitions
        valid = set(self.next_valid())

        def has_dependency(word1, word2):
            """  check if there's any dependency between the 2 words """
            return gold_heads[word1] == word2 or gold_heads[word2] == word1

        if not self.stack or (
                        Transitions.SHIFT in valid and
                        gold_heads[self.buffer_index] == self.stack[-1]):
            return [Transitions.SHIFT]

        # If there's an edge between the node on top of the
        # stack and the current index in buffer, we have no choice but to
        # draw that edge, otherwise we'll loose it
        if gold_heads[self.stack[-1]] == self.buffer_index:
            return [Transitions.LEFT_ARC]

        # If stack[-2] is the parent of stack[-1] then drawing an edge from
        # stack[-1] to stack[-2] is incorrect (wrong direction)
        if len(self.stack) >= 2 and gold_heads[self.stack[-1]] == self.stack[-2]:
            valid.discard(Transitions.LEFT_ARC)

        # If there's any dependency between the current item in the buffer and
        # a node in the stack, moving the item in the stack would loose the dependency
        if any([has_dependency(self.buffer_index, w) for w in self.stack]):
            valid.discard(Transitions.SHIFT)

        # If there's any dependency between stack[-1] and a node in the buffer,
        # popping the stack would loose that dependency
        if any([has_dependency(self.stack[-1], w) for w in buffer_nodes]):
            valid.discard(Transitions.LEFT_ARC)
            valid.discard(Transitions.RIGHT_ARC)

        return list(valid)


class Parser(ParserI):

    @staticmethod
    def build_labels_dataset(parses, feature_extractor):
        """ Transform a list of parses to a labels dataset """
        labels_X, labels_y = [], []
        for gold_parse in parses:
            for child, head in enumerate(gold_parse.heads()[1:-1]):
                features = feature_extractor(gold_parse, head, child + 1)

                label = gold_parse.labels()[child + 1]
                labels_X.append(features)
                labels_y.append(label)

        return labels_X, labels_y

    @staticmethod
    def build_transition_dataset(parses, feature_extractor):
        """ Transform a list of parses to a transitions dataset """
        transitions_X, transitions_y = [], []
        for gold_parse in parses:
            # Init an empty parse
            dep_parse = DependencyParse(gold_parse.tagged_words()[1:-1])

            # Start from an empty state
            state = ParserState(dep_parse)

            while state.stack or (state.buffer_index + 1) < len(dep_parse):
                features = feature_extractor(state)
                gold_moves = state.next_gold(gold_parse)

                if not gold_moves:
                    # Something is wrong here ...
                    break

                # Pick one of the possible transitions
                t = random.choice(gold_moves)

                # Append the features and transition to the dataset
                transitions_X.append(features)
                transitions_y.append(t)

                # Apply the transition to the state
                state.apply(t)

        return transitions_X, transitions_y

    def __init__(self, feature_detector, label_feature_detector):
        self.feature_extractor = feature_detector
        self.label_feature_detector = label_feature_detector

        self._vectorizer = FeatureHasher()
        self._model = SGDClassifier(loss='modified_huber')

        self._label_vectorizer = FeatureHasher()
        self._label_model = Perceptron()

    def evaluate(self, parses):
        correct_heads, correct_labels, total = 0, 0, 0

        for parse in parses:
            predicted_parse = self.parse(parse.tagged_words()[1:-1])

            heads = np.array(parse.heads()[1:-1])
            predicted_heads = np.array(predicted_parse.heads()[1:-1])

            labels = np.array(parse.labels()[1:-1])

            # Relabel the gold parse with what our model would label
            self.label_parse(parse)
            predicted_labels = np.array(parse.labels()[1:-1])

            total += len(heads)
            correct_heads += np.sum(heads == predicted_heads)
            correct_labels += np.sum(labels == predicted_labels)

        return correct_heads / total, correct_labels / total

    def parse(self, sent, *args, **kwargs):
        """ Parse a tagged sentence """
        state = ParserState(DependencyParse(sent))
        while state.stack or (state.buffer_index + 1) < len(state.parse):
            # Extract the features of the current state
            features = self.feature_extractor(state)
            vectorized_features = self._vectorizer.transform([features])

            # Get probabilities for the next transitions
            predictions = self._model.predict_proba(vectorized_features)[0]
            scores = dict(zip(list(self._model.classes_), list(predictions)))

            # Check what moves are actually valid
            valid_moves = state.next_valid()

            # Get the most probable valid mode
            guess = max(valid_moves, key=lambda move: scores[move])

            # apply the transition to the state
            state.apply(guess)

        self.label_parse(state.parse)  # Add labels too ...

        return state.parse

    def label_parse(self, parse):
        """ Add labels to a dependency parse """
        label_features = []
        for child, head in enumerate(parse.heads()[1:-1]):
            features = self.label_feature_detector(parse, head, child + 1)
            label_features.append(features)

        vectorized_label_features = self._label_vectorizer.transform(label_features)
        predicted_labels = self._label_model.predict(vectorized_label_features)
        parse._labels = [None] + list(predicted_labels) + [None]

        return parse

    def train(self, corpus_iterator, n_iter=5, batch_size=100):
        """ Train a model on a given corpus """
        for _ in range(n_iter):
            # Fork the iterator
            corpus_iterator, parses = itertools.tee(corpus_iterator)
            batch_count = 0

            while True:
                batch_count += 1
                print("Training on batch={0}".format(batch_count))
                batch = list(itertools.islice(parses, batch_size))

                # No more batches
                if not batch:
                    break

                # Train the model on a batch
                self.train_batch(batch)

    def train_batch(self, gold_parses):
        """ Train the model on a single batch """
        t_X, t_Y = self.build_transition_dataset(
            gold_parses, self.feature_extractor)

        self._model.partial_fit(self._vectorizer.transform(t_X), t_Y,
                                classes=Transitions.ALL)

        l_X, l_Y = self.build_labels_dataset(
            gold_parses, self.label_feature_detector)

        self._label_model.partial_fit(self._label_vectorizer.transform(l_X), l_Y,
                                      classes=DEPENDENCY_LABELS)


def main():
    train_data = list(iter_ud_dependencies('../../../data/en-ud-train.conllu'))
    test_data = list(iter_ud_dependencies('../../../data/en-ud-dev.conllu'))

    parser = Parser(dependency_features, edge_label_features)
    parser.train(train_data, n_iter=5, batch_size=200)

    print("Accuracy: ", parser.evaluate(test_data[:250]))

    p = parser.parse(pos_tag(word_tokenize("I buy green apples")))
    print(p.heads())
    print(p.labels())


if __name__ == '__main__':
    main()