from utils import pad_right, iter_ud_dep_data


class DependencyParse(object):
    def __init__(self, tagged_sentence):
        # pad the sentence with a dummy __START__ and a __ROOT__ node
        self._tagged = [('__START__', '__START__')] + tagged_sentence + [('__ROOT__', '__ROOT__')]

        # parent list and labels
        self._heads = [None] * len(self)
        self._labels = [None] * len(self)

        # keep a list of dependents and the labels
        self._deps = [[] for _ in range(len(self))]

        # keep a list of left dependants and the labels
        self._left_deps = [[] for _ in range(len(self))]

        # keep a list of the right dependants and the labels
        self._right_deps = [[] for _ in range(len(self))]


    def add_edge(self, head, child, label=None):
        """ Add labelled edge between 2 nodes in the dependency graph """
        self._heads[child] = head
        self._labels[child] = label

        # keep track of all dependents of a node
        self._deps[head].append(child)

        # keep track of the left/rights dependents
        if child < head:
            self._left_deps[head].append(child)
        else:
            self._right_deps[head].append(child)

    def words(self):
        return [tt[0] for tt in self._tagged]

    def tags(self):
        return [tt[1] for tt in self._tagged]

    def tagged_words(self):
        return self._tagged

    def heads(self):
        return self._heads

    def lefts(self):
        return self._left_deps

    def rights(self):
        return self._right_deps

    def deps(self):
        return self._deps

    def labels(self):
        return self._labels

    def __getitem__(self, item):
        return {
            'word': self._tagged[item][0],
            'pos': self._tagged[item][1],
            'head': self._heads[item],
            'label': self._labels[item],
            'children': list(zip(self._deps[item], self._labels[item])),
            'left_children': self._left_deps[item],
            'right_children': self._right_deps[item],
        }

    def __len__(self):
        return len(self._tagged)

    def __str__(self):
        return str(self._heads)

    def __repr__(self):
        return str(self._heads)


def dependency_features(state):
    """ Extract features from a parser state """
    length = len(state.parse)
    words = state.parse.words()
    tags = state.parse.tags()

    def stack_features(data, width=3):
        context = [data[idx] for idx in reversed(state.stack[-width:])]
        return tuple(pad_right(context, width, '__NONE__'))

    def buffer_features(data, width=3):
        context = [data[idx] for idx in
                   range(state.buffer_index, min(state.buffer_index + width, length))]
        return tuple(pad_right(context, width, '__NONE__'))

    def parse_features(dependencies, data, width=2):
        context = [data[idx] for idx in reversed(dependencies[-width:])]
        return tuple(pad_right(context, width, '__NONE__'))

    f = {}

    # Top node in stack
    top_s = state.stack[-1] if len(state.stack) else -1

    # stack - words/tags
    f['w-s0'], f['w-s1'], f['w-s2'] = stack_features(words)
    f['t-s0'], f['t-s1'], f['t-s2'] = stack_features(tags)

    # buffer - words/tags
    f['w-b0'], f['w-b1'], f['w-b2'] = buffer_features(words)
    f['t-b0'], f['t-b1'], f['t-b2'] = buffer_features(tags)

    # =============================================

    # left children - buffer - words
    f['w-lp-b0'], f['w-lp-b1'] = parse_features(
        state.parse.lefts()[state.buffer_index], words)

    # left children - buffer - tags
    f['t-lp-b0'], f['t-lp-b1'] = parse_features(
        state.parse.lefts()[state.buffer_index], tags)

    # left children - buffer - count
    f['#lp-b'] = len(state.parse.lefts()[state.buffer_index])

    # =============================================

    # right children - buffer - words
    f['w-rp-b0'], f['w-rp-b1'] = parse_features(
        state.parse.rights()[state.buffer_index], words)

    # right children - buffer - tags
    f['t-rp-b0'], f['t-rp-b1'] = parse_features(
        state.parse.rights()[state.buffer_index], tags)

    # right children - buffer - count
    f['#rp-b'] = len(state.parse.rights()[state.buffer_index])

    # =============================================

    # left children - stack - words
    f['w-lp-s0'], f['w-lp-s1'] = parse_features(
        state.parse.lefts()[top_s], words)

    # left children - stack - tags
    f['t-lp-s0'], f['t-lp-s1'] = parse_features(
        state.parse.lefts()[top_s], tags)

    # left children - stack - count
    f['#lp-s'] = len(state.parse.lefts()[top_s])

    # =============================================

    # right children - stack - words
    f['w-rp-s0'], f['w-rp-s1'] = parse_features(
        state.parse.rights()[top_s], words)

    # right children - stack - tags
    f['t-rp-s0'], f['t-rp-s1'] = parse_features(
        state.parse.rights()[top_s], tags)

    # right children - stack - count
    f['#rp-s'] = len(state.parse.rights()[top_s])

    # =============================================

    # distance between the top node in the stack and the current node in the buffer
    f['dist-b-s'] = state.buffer_index - top_s

    # =============================================

    # Word/Tag pairs
    f['w-s0$t-s0'] = f['w-s0'] + '$' + f['t-s0']
    f['w-s1$t-s1'] = f['w-s1'] + '$' + f['t-s1']
    f['w-s2$t-s2'] = f['w-s2'] + '$' + f['t-s2']

    f['w-b0$t-b0'] = f['w-b0'] + '$' + f['t-b0']
    f['w-b1$t-b1'] = f['w-b1'] + '$' + f['t-b1']
    f['w-b2$t-b2'] = f['w-b2'] + '$' + f['t-b2']

    # =============================================

    # Bigrams
    f['w-s0$w-b0'] = f['w-s0'] + '$' + f['w-b0']
    f['t-s0$t-b0'] = f['t-s0'] + '$' + f['t-b0']

    f['w-s1$w-b1'] = f['w-s1'] + '$' + f['w-b1']
    f['t-s1$t-b1'] = f['t-s1'] + '$' + f['t-b1']

    f['w-s2$w-b2'] = f['w-s2'] + '$' + f['w-b2']
    f['t-s2$t-b2'] = f['t-s2'] + '$' + f['t-b2']

    f['w-s0$w-s1'] = f['w-s0'] + '$' + f['w-s1']
    f['w-s1$w-s2'] = f['w-s1'] + '$' + f['w-s2']

    f['t-s0$t-s1'] = f['t-s0'] + '$' + f['t-s1']
    f['t-s1$t-s2'] = f['t-s1'] + '$' + f['t-s2']

    f['w-b0$w-b1'] = f['w-b0'] + '$' + f['w-b1']
    f['w-b1$w-b2'] = f['w-b1'] + '$' + f['w-b2']

    f['t-b0$t-b1'] = f['t-b0'] + '$' + f['t-b1']
    f['t-b1$t-b2'] = f['t-b1'] + '$' + f['t-b2']

    # =============================================

    # Trigrams
    f['w-s0$w-s1$w-s2'] = f['w-s0$w-s1'] + '$' + f['w-s2']
    f['t-s0$t-s1$t-s2'] = f['t-s0$t-s1'] + '$' + f['t-s2']

    f['w-b0$w-b1$w-b2'] = f['w-b0$w-b1'] + '$' + f['w-b2']
    f['t-b0$t-b1$t-b2'] = f['t-b0$t-b1'] + '$' + f['t-b2']

    f['w-s0$w-lp-s0$w-lp-s1'] = f['w-s0'] + '$' + f['w-lp-s0'] + '$' + f['w-lp-s1']
    f['t-s0$t-lp-s0$t-lp-s1'] = f['t-s0'] + '$' + f['t-lp-s0'] + '$' + f['t-lp-s1']

    f['w-b0$w-lp-b0$w-lp-b1'] = f['w-b0'] + '$' + f['w-lp-b0'] + '$' + f['w-lp-b1']
    f['t-b0$t-lp-b0$t-lp-b1'] = f['t-b0'] + '$' + f['t-lp-b0'] + '$' + f['t-lp-b1']

    return f


def iter_ud_dependencies(filename):
    for annotated_sentence in iter_ud_dep_data(filename):
        words = []
        tags = []

        heads = [None]
        labels = [None]
        for i, (word, pos, head, label) in enumerate(annotated_sentence):
            # Skip some fictive nodes that don't belong in the graph
            if head == '_':
                continue

            words.append(word)
            tags.append(pos)

            if head == '0':
                # Point to the dummy node
                heads.append(len(annotated_sentence) + 1)
            else:
                heads.append(int(head))

            labels.append(label)

        dep_parse = DependencyParse(list(zip(words, tags)))
        for child, head in enumerate(heads):
            if head is not None:
                dep_parse.add_edge(head, child, labels[child])

        yield dep_parse


def edge_label_features(parse, head, child):
    f = {
        'head-pos': parse.tags()[head],
        'child-pos': parse.tags()[child],
        'head-word': parse.words()[head],
        'child-word': parse.words()[child],
        'distance': head - child,
        'head-before': head < child,

        'head-1-word': parse.words()[head - 1],
        'head-1-pos': parse.tags()[head - 1],
        'head+1-word': parse.words()[head + 1]
            if head + 1 < len(parse) else '__NONE__',
        'head+1-pos': parse.tags()[head + 1]
            if head + 1 < len(parse) else '__NONE__',

        'child-1-word': parse.words()[child - 1],
        'child-1-pos': parse.tags()[child - 1],
        'child+1-word': parse.words()[child + 1],
        'child+1-pos': parse.tags()[child + 1],
    }

    f['child-word$head-word'] = f['child-word'] + "$" + f['head-word']
    f['child-pos$head-pos'] = f['child-pos'] + "$" + f['head-pos']

    f['child-1-pos$child-pos'] = f['child-1-pos'] + "$" + f['child-pos']
    f['child-pos$child+1-pos'] = f['child-pos'] + "$" + f['child+1-pos']
    f['child-1-word$child-word'] = f['child-1-word'] + "$" + f['child-word']
    f['child-word$child+1-word'] = f['child-word'] + "$" + f['child+1-word']

    f['head-1-pos$head-pos'] = f['head-1-pos'] + "$" + f['head-pos']
    f['head-pos$head+1-pos'] = f['head-pos'] + "$" + f['head+1-pos']
    f['head-1-word$head-word'] = f['head-1-word'] + "$" + f['head-word']
    f['head-word$head+1-word'] = f['head-word'] + "$" + f['head+1-word']

    f['child-1-pos$child-pos$child+1-pos'] = f['child-1-pos$child-pos'] + "$" + f['child+1-pos']
    f['child-1-word$child-word$child+1-word'] = f['child-1-word$child-word'] + "$" + f['child+1-word']

    f['head-1-pos$head-pos$head+1-pos'] = f['head-1-pos$head-pos'] + "$" + f['head+1-pos']
    f['head-1-word$head-word$head+1-word'] = f['head-1-word$head-word'] + "$" + f['head+1-word']

    return f