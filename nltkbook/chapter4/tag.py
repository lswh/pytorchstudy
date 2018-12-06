from nltk.stem.snowball import SnowballStemmer
from features import shape


stemmer = SnowballStemmer('english')


def pos_features(sentence, index, history):
    """
    sentence = list of words: [word1, word2, ...]
    index = the index of the word we want to extract features for
    history = the list of predicted tags of the previous tokens
    """
    # Pad the sequence with placeholders
    # We will be looking at two words back and forward, so need to make sure we do not go out of bounds
    sentence = ['__START2__', '__START1__'] + list(sentence) + ['__END1__', '__END2__']

    # We will be looking two words back in history, so need to make sure we do not go out of bounds
    history = ['__START2__', '__START1__'] + list(history)

    # shift the index with 2, to accommodate the padding
    index += 2

    return {
        # Intrinsic features
        'word': sentence[index],
        'stem': stemmer.stem(sentence[index]),
        'shape': shape(sentence[index]),

        # Suffixes
        'suffix-1': sentence[index][-1],
        'suffix-2': sentence[index][-2:],
        'suffix-3': sentence[index][-3:],

        # Context
        'prev-word': sentence[index - 1],
        'prev-stem': stemmer.stem(sentence[index - 1]),
        'prev-prev-word': sentence[index - 2],
        'prev-prev-stem': stemmer.stem(sentence[index - 2]),
        'next-word': sentence[index + 1],
        'next-stem': stemmer.stem(sentence[index + 1]),
        'next-next-word': sentence[index + 2],
        'next-next-stem': stemmer.stem(sentence[index + 2]),

        # Historical features
        'prev-pos': history[-1],
        'prev-prev-pos': history[-2],

        # Composite
        'prev-word+word': sentence[index - 1].lower() + '+' + sentence[index],
    }