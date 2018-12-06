import os
from nltk import conlltags2tree


def read_ud_pos_data(filename):
    """
    Iterate through the Universal Dependencies Corpus Part-Of-Speech data
    Yield sentences one by one, don't load all the data in memory
    """
    current_sentence = []

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()

            # ignore comments
            if line.startswith('#'):
                continue

            # empty line indicates end of sentence
            if not line:
                yield current_sentence

                current_sentence = []
                continue

            annotations = line.split('\t')

            # Get only the word and the part of speech
            current_sentence.append((annotations[1], annotations[4]))


def ner2conlliob(annotated_sentence):
    """
    Transform the pseudo NER annotated sentence to proper IOB annotation

    Example:
    [(word1, pos1, O), (word2, pos2, PERSON), (word3, pos3, PERSON),
     (word4, pos4, O), (word5, pos5, O), (word6, pos6, LOCATION)]

    transforms to:

    [(word1, pos1, O), (word2, pos2, B-PERSON), (word3, pos3, I-PERSON),
     (word4, pos4, O), (word5, pos5, O), (word6, pos6, B-LOCATION)]
    """
    iob_tokens = []
    for idx, annotated_token in enumerate(annotated_sentence):
        tag, word, ner = annotated_token

        if ner != 'O':
            # If it's the first NE is also the first word
            if idx == 0:
                ner = "B-{0}".format(ner)

            # If the previous NE token is the same as the current one
            elif annotated_sentence[idx - 1][2] == ner:
                ner = "I-{0}".format(ner)

            # The NE sequence just started
            else:
                ner = "B-{0}".format(ner)

        iob_tokens.append((tag, word, ner))
    return iob_tokens


def read_gmb_ner(corpus_root, start_index=None, end_index=None):
    current_file = -1
    for root, _, files in os.walk(corpus_root):
        for filename in files:
            # Skip other files
            if not filename.endswith(".tags"):
                continue

            current_file += 1
            # Skip files until we get to the start_index
            if start_index is not None and current_file < start_index:
                continue

            # Stop reading after end_index
            if end_index is not None and current_file > end_index:
                return

            with open(os.path.join(root, filename), 'rb') as file_handle:
                # Read the entire file
                file_content = file_handle.read().decode('utf-8').strip()

                # Split into sentences
                annotated_sentences = file_content.split('\n\n')

                for annotated_sentence in annotated_sentences:
                    # Split into annotated tokens
                    rows = [row for row in annotated_sentence.split('\n') if row]

                    ner_triplets = []
                    for row in rows:
                        annotations = row.split('\t')
                        word, tag, ner = annotations[0], annotations[1], annotations[3]

                        # Get only the main tag
                        if ner != 'O':
                            ner = ner.split('-')[0]

                        # Make these tags NLTK compatible
                        if tag in ('LQU', 'RQU'):
                            tag = "``"

                        # Ignore the art,eve,nat tags because they are underrepresented
                        if tag in ('art', 'eve', 'nat'):
                            tag = 'O'

                        ner_triplets.append((word, tag, ner))

                    iob_triplets = ner2conlliob(ner_triplets)

                    # Yield a nltk.Tree
                    yield conlltags2tree(iob_triplets)
    print("Total files=", current_file)


def pad_right(l, n, padding=None):
    if n <= len(l):
        return l
    return l + ([padding] * (n - len(l)))


def iter_ud_dep_data(filename):
    current_sentence = []

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()

            # ignore comments
            if line.startswith('#'):
                continue

            # empty line indicates end of sentence
            if not line:
                yield current_sentence

                current_sentence = []
                continue

            annotations = line.split('\t')

            # Get only the word, pos, head and dep rel
            try:
                int(annotations[0])
                current_sentence.append(
                    (annotations[1], annotations[4], annotations[6], annotations[7])
                )
            except ValueError:
                pass
