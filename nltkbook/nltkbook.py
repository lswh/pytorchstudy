import nltk
from nltk.corpus import reuters
sentences = nltk.sent_tokenize(reuters.raw('test/21131')[:1000]) 
print("#sentences={0}\n\n".format(len(sentences)))
for sent in sentences:
    print(sent, '\n')
