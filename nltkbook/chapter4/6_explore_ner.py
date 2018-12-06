from nltk import pos_tag, ne_chunk


print(ne_chunk(pos_tag(
    "Twitter Inc. is based in San Francisco , California , United States , "
    "and has more than 25 offices around the world .".split())))


# (S
#   (PERSON Twitter/NNP)
#   (ORGANIZATION Inc./NNP)
#   is/VBZ
#   based/VBN
#   in/IN
#   (GPE San/NNP Francisco/NNP)
#   ,/,
#   (GPE California/NNP)
#   ,/,
#   (GPE United/NNP States/NNPS)
#   ,/,
#   and/CC
#   has/VBZ
#   more/JJR
#   than/IN
#   25/CD
#   offices/NNS
#   around/IN
#   the/DT
#   world/NN
#   ./.)