import numpy as np


def clean_and_concat(dirty_string: str) -> str:
    '''
    Clearing a string of characters
    that are not numbers or letters

    Args:
        dirty_string: 'Dirty' string

    Returns:
        String contains only letters and numbers
    '''
    res = ''
    for char in dirty_string:
        if str.isalnum(char):
            res += char

    return res


def transliterate(text: str) -> str:
    '''
    Transliteration from Russian to English letters

    Args:
        text: Text to be transliterated

    Returns:
        Text in which all Russian letters are replaced
        with their English counterparts
    '''

    # List of all Russian letters
    rus = list('ёйцукенгшщзхъфывапролджэячсмитьбю')

    res = ''

    # Matching of Russian and English letters
    mapper = {
        'ё': 'yo',
        'й': 'y',
        'ц': 'ts',
        'у': 'u',
        'к': 'k',
        'е': 'e',
        'н': 'n',
        'г': 'g',
        'ш': 'sh',
        'щ': 'shch',
        'з': 'z',
        'х': 'kh',
        'ъ': '',
        'ф': 'f',
        'ы': 'y',
        'в': 'v',
        'а': 'a',
        'п': 'p',
        'р': 'r',
        'о': 'o',
        'л': 'l',
        'д': 'd',
        'ж': 'zh',
        'э': 'e',
        'я': 'ya',
        'ч': 'ch',
        'с': 's',
        'м': 'm',
        'и': 'i',
        'т': 't',
        'ь': '',
        'б': 'b',
        'ю': 'yu'
    }

    if np.isin(list(text), rus).any():
        res = ''.join(
            [mapper.get(char, char) for char in text]
        )
        return res

    # If text doesn't contain Russian
    # letters return original text
    return text
