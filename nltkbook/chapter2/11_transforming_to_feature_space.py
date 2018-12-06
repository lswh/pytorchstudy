import collections

text = """
How much wood does a woodchuck chuck
if a woodchuck could chuck wood
"""

print(collections.Counter(text.lower().split()))
# Counter({
#   'wood': 2,
#   'a': 2,
#   'woodchuck': 2,
#   'chuck': 2,
#   'How': 1,
#   'much': 1,
#   'does': 1,
#   'if': 1,
#   'could': 1
# })

{
    'wood': 2,
    'a': 2,
    'woodchuck': 2,
    'chuck': 2,
    'wow': 1,
    'much': 1,
    'does': 1,
    'if': 1,
    'could': 1
}