import string


def extract_features(name):
    """
    Get the features used for name classification
    """
    features = {
        # Last letter
        'last_letter': name[-1],
        # First letter
        'first_letter': name[0],
        # How many vowels
        'vowel_count': len([c for c in name if c in 'AEIOUaeiou'])
    }
    # Build letter and letter count features
    for c in string.ascii_lowercase:
        features['contains_' + c] = c in name
        features['count_' + c] = name.lower().count(c)
    return features