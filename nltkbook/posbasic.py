import nltk
sentence = "Things I wish I knew before I started blogging." 
tokens = nltk.word_tokenize(sentence)
print("Tokens: ", tokens)
# Tokens: ['Things', 'I', 'wish', 'I', 'knew', 'before', 'I', 'started', 'blogging', '.']
tagged_tokens = nltk.pos_tag(tokens) 
print("Tagged Tokens: ", tagged_tokens)
# Tagged Tokens: [('Things', 'NNS'), ('I', 'PRP'), ('wish', 'VBP'), ('I', 'PRP'), ('knew', 'VBD'), ...
