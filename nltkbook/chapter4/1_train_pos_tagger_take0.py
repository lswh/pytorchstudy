import nltk
import time
from collections import Counter
from utils import read_ud_pos_data

# Compute the most common tag
tag_counter = Counter()
train_data = read_ud_pos_data('../../../data/en-ud-train.conllu')
for sentence in train_data:
    tag_counter.update([t for _, t in sentence])

# Peek at what are the most common 5 tags
print(tag_counter.most_common(5))
# [('NN', 26915), ('IN', 20724), ('DT', 16817), ('NNP', 12449), ('PRP', 12193)]

most_common_tag = tag_counter.most_common()[0][0]
print("Most Common Tag is: ", most_common_tag) # NN

# Load the data for training
train_data = read_ud_pos_data('../../../data/en-ud-train.conllu')

# Load the data for testing
test_data = read_ud_pos_data('../../../data/en-ud-dev.conllu')

start_time = time.time()
print("Starting training ...")

t0 = nltk.DefaultTagger('NN')
t1 = nltk.UnigramTagger(train_data, backoff=t0)
t2 = nltk.BigramTagger(train_data, backoff=t1)
t3 = nltk.TrigramTagger(train_data, backoff=t2)

end_time = time.time()
print("Training complete. Time={0:.2f}s".format(end_time - start_time))

# Compute test set accuracy
print(t3.evaluate(list(test_data)))  # 0.8391919834579291

# Here's how to use our new tagger
print(t3.tag("This is a test".split()))