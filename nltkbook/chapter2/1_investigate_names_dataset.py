import collections
from nltk.corpus import names


# Check the girl names
girl_names = names.words('female.txt')
print(girl_names[:10], '...')
print('#GirlNames=', len(girl_names))  # 5001

# Check the boy names
boy_names = names.words('male.txt')
print(boy_names[:10], '...')
print('#BoyNames=', len(boy_names))  # 2943

# We know that there are a lot of girl names that end in `a`
# Let's see how many
girl_names_ending_in_a = [name for name in girl_names if name.endswith('a')]
print('#GirlNamesEndingInA=', len(girl_names_ending_in_a))  # 1773

# Approx. a third of girl names end in `a`. That's a good insight
# Let's see what are the most common letters girl names end with
girl_ending_letters = collections.Counter([name[-1] for name in girl_names])
print("MostCommonEndingLettersForGirls=", girl_ending_letters)

# Our intuition was right. The most common letter is `a`
# Here are the first 3: 'a': 1773, 'e': 1432, 'y': 461

# I'm not sure what's the most common last letter for English boy names
# Let's see the stats
boy_ending_letters = collections.Counter([name[-1] for name in boy_names])
print("MostCommonEndingLettersForBoys=", boy_ending_letters)
# Here are the first 3: 'n': 478, 'e': 468, 'y': 332,
