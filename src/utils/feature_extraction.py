from collections import Counter
import math
import re

def text_to_vector(text: str) -> Counter:
    '''
    Transform text to vector

    Args:
        text: Original text

    Returns:
        Corresponding vector
    '''
    word = re.compile(r'\w+')
    words = word.findall(text)

    return Counter(words)


def cos_distance(string_1: str, string_2: str) -> float:
    '''
    Get cosine distance between two strings

    Args:
            string_1: First string
            string_2: Second string

    Returns:
        Cosine distance between vectors
    '''
    def _get_cosine(vec_1: Counter, vec_2: Counter) -> float:
        '''
        Get cosine distance between two vectors

        Args:
            vec_1: First vector
            vec_2: Second vector

        Returns:
            Cosine distance between vectors
        '''
        intersection = set(vec_1.keys()) & set(vec_2.keys())
        numerator = sum([vec_1[x] * vec_2[x] for x in intersection])

        sum1 = sum([vec_1[x]**2 for x in vec_2.keys()])
        sum2 = sum([vec_1[x]**2 for x in vec_2.keys()])

        denominator = math.sqrt(sum1) * math.sqrt(sum2)

        if not denominator:
            return 0.0
        else:
            return float(numerator) / denominator

    vector_1 = text_to_vector(string_1)
    vector_2 = text_to_vector(string_2)

    return _get_cosine(vector_1, vector_2)


def jaccard_similarity(string_1: str, string_2: str) -> float:
    '''
    Get jaccard similarity of two strings

    Args:
      string_1: First string
      string_2: Second string

    Returns:
        Jaccard similarity of string
    '''
    intersection_cardinality = len(
        set.intersection(*[set(string_1), set(string_2)]))
    union_cardinality = len(set.union(*[set(string_1), set(string_2)]))

    return intersection_cardinality / float(union_cardinality)
