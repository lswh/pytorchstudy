from nltk.corpus import conll2000

print(len(conll2000.chunked_sents()))  # 10948
print(len(conll2000.chunked_words()))  # 166433

chunked_sentence = conll2000.chunked_sents()[0]
print(chunked_sentence)

# (S
#   (NP Confidence/NN)
#   (PP in/IN)
#   (NP the/DT pound/NN)
#   (VP is/VBZ widely/RB expected/VBN to/TO take/VB)
#   (NP another/DT sharp/JJ dive/NN)
#   if/IN
#   (NP trade/NN figures/NNS)
#   (PP for/IN)
#   (NP September/NNP)
#   ,/,
#   due/JJ
#   (PP for/IN)
#   (NP release/NN)
#   (NP tomorrow/NN)
#   ,/,
#   (VP fail/VB to/TO show/VB)
#   (NP a/DT substantial/JJ improvement/NN)
#   (PP from/IN)
#   (NP July/NNP and/CC August/NNP)
#   (NP 's/POS near-record/JJ deficits/NNS)
#   ./.)

from nltk.chunk import tree2conlltags
iob_tagged = tree2conlltags(chunked_sentence)
print(iob_tagged)

# [
#   ('Confidence', 'NN', 'B-NP'),
#   ('in', 'IN', 'B-PP'),
#   ('the', 'DT', 'B-NP'),
#   ('pound', 'NN', 'I-NP'),
#   ('is', 'VBZ', 'B-VP'),
#   ('widely', 'RB', 'I-VP'),
#   ('expected', 'VBN', 'I-VP'),
#   ('to', 'TO', 'I-VP'),
#   ('take', 'VB', 'I-VP'),
#   ('another', 'DT', 'B-NP'),
#   ('sharp', 'JJ', 'I-NP'),
#   ('dive', 'NN', 'I-NP'),
#   ('if', 'IN', 'O'),
#   ('trade', 'NN', 'B-NP'),
#   ('figures', 'NNS', 'I-NP'),
#   ('for', 'IN', 'B-PP'),
#   ('September', 'NNP', 'B-NP'),
#   (',', ',', 'O'),
#   ('due', 'JJ', 'O'),
#   ('for', 'IN', 'B-PP'),
#   ('release', 'NN', 'B-NP'),
#   ('tomorrow', 'NN', 'B-NP'),
#   (',', ',', 'O'),
#   ('fail', 'VB', 'B-VP'),
#   ('to', 'TO', 'I-VP'),
#   ('show', 'VB', 'I-VP'),
#   ('a', 'DT', 'B-NP'),
#   ('substantial', 'JJ', 'I-NP'),
#   ('improvement', 'NN', 'I-NP'),
#   ('from', 'IN', 'B-PP'),
#   ('July', 'NNP', 'B-NP'),
#   ('and', 'CC', 'I-NP'),
#   ('August', 'NNP', 'I-NP'),
#   ("'s", 'POS', 'B-NP'),
#   ('near-record', 'JJ', 'I-NP'),
#   ('deficits', 'NNS', 'I-NP'),
#   ('.', '.', 'O')
# ]

from nltk.chunk import conlltags2tree
chunk_tree = conlltags2tree(iob_tagged)
print(chunk_tree)

# (S
#   (NP Confidence/NN)
#   (PP in/IN)
#   (NP the/DT pound/NN)
#   (VP is/VBZ widely/RB expected/VBN to/TO take/VB)
#   (NP another/DT sharp/JJ dive/NN)
#   if/IN
#   (NP trade/NN figures/NNS)
#   (PP for/IN)
#   (NP September/NNP)
#   ,/,
#   due/JJ
#   (PP for/IN)
#   (NP release/NN)
#   (NP tomorrow/NN)
#   ,/,
#   (VP fail/VB to/TO show/VB)
#   (NP a/DT substantial/JJ improvement/NN)
#   (PP from/IN)
#   (NP July/NNP and/CC August/NNP)
#   (NP 's/POS near-record/JJ deficits/NNS)
#   ./.)