def extract_features(name):
    """
    Get the features used for name classification
    """
    return {
        'last_letter': name[-1],
        'vowel_count': len([c for c in name if c in 'AEIOUaeiou'])
    }