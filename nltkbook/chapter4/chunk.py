from nltk import ChunkParserI, tree2conlltags, conlltags2tree
from classify import ClassifierBasedTaggerBatchTrained
from nltk.metrics import accuracy


def triplets2tagged_pairs(iob_sent):
    """
        Transform the triplets to tagged pairs:
        [(word1, pos1, iob1), (word2, pos2, iob2), ...] ->
        [((word1, pos1), iob1), ((word2, pos2), iob2),...]
    """
    return [((word, pos), chunk) for word, pos, chunk in iob_sent]


def tagged_pairs2triplets(iob_sent):
    """
        Transform the triplets to tagged pairs:
        [((word1, pos1), iob1), ((word2, pos2), iob2),...] ->
        [(word1, pos1, iob1), (word2, pos2, iob2), ...]
    """
    return [(word, pos, chunk) for (word, pos), chunk in iob_sent]


class ClassifierBasedChunkParser(ChunkParserI):
    def __init__(self, chunked_sents, feature_detector, classifier_builder, **kwargs):
        # Transform the trees in IOB annotated sentences [(word, pos, chunk), ...]
        chunked_sents = [tree2conlltags(sent) for sent in chunked_sents]

        chunked_sents = [triplets2tagged_pairs(sent) for sent in chunked_sents]

        self.feature_detector = feature_detector

        self.tagger = ClassifierBasedTaggerBatchTrained(
            train=(sent for sent in chunked_sents),
            feature_detector=self.feature_detector,
            classifier_builder=classifier_builder
        )

    def parse(self, tagged_sent):
        chunks = self.tagger.tag(tagged_sent)

        iob_triplets = tagged_pairs2triplets(chunks)

        # Transform the list of triplets to nltk.Tree format
        return conlltags2tree(iob_triplets)

    def evaluate(self, gold):
        # Convert nltk.Tree chunked sentences to (word, pos, iob) triplets
        chunked_sents = [tree2conlltags(sent) for sent in gold]

        # Convert (word, pos, iob) triplets to tagged tuples ((word, pos), iob)
        chunked_sents = [triplets2tagged_pairs(sent) for sent in chunked_sents]

        print(chunked_sents)

        dataset = self.tagger._todataset(chunked_sents)
        featuresets, tags = zip(*dataset)
        predicted_tags = self.tagger.classifier().classify_many(featuresets)
        return accuracy(tags, predicted_tags)


from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer('english')


def chunk_features(tokens, index, history):
    """
    `tokens`  = a POS-tagged sentence [(w1, t1), ...]
    `index`   = the index of the token we want to extract features for
    `history` = the previous predicted IOB tags
    """

    # Pad the sequence with placeholders
    tokens = ([('__START2__', '__START2__'), ('__START1__', '__START1__')] + 
              list(tokens) + 
              [('__END1__', '__END1__'), ('__END2__', '__END2__')])
    history = ['__START2__', '__START1__'] + list(history)

    # shift the index with 2, to accommodate the padding
    index += 2

    word, pos = tokens[index]
    prevword, prevpos = tokens[index - 1]
    prevprevword, prevprevpos = tokens[index - 2]
    nextword, nextpos = tokens[index + 1]
    nextnextword, nextnextpos = tokens[index + 2]

    return {
        'word': word,
        'lemma': stemmer.stem(word),
        'pos': pos,

        'next-word': nextword,
        'next-pos': nextpos,

        'next-next-word': nextnextword,
        'nextnextpos': nextnextpos,

        'prev-word': prevword,
        'prev-pos': prevpos,

        'prev-prev-word': prevprevword,
        'prev-prev-pos': prevprevpos,

        # Historical features
        'prev-chunk': history[-1],
        'prev-prev-chunk': history[-2],
    }