import nltk
# Adapted from Wikipedia:
# https://en.wikipedia.org/wiki/Surely_You%27re_Joking,_Mr._Feynman!
sentence = "The closing chapter, is adapted from the address that Feynman gave during the 1974 commencement exercises at the California Institute Of Technology."
# tokenize and pos tag
tokens = nltk.word_tokenize(sentence) 
tagged_tokens = nltk.pos_tag(tokens) 
ner_annotated_tree = nltk.ne_chunk(tagged_tokens)
print(ner_annotated_tree)
