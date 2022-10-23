from typing import List

from transliterate import translit

DROP_SYMBOLS = ['#', '%', '&', '*', '+', ',', '.', '/', ';', '<', '>', '?',
                '@', '[', '\\', ']', '`', '{', '\xa0', '«', '±', '»', '¿',
                'ر', 'س', 'ف', 'ك', 'م', 'و', 'ي', '\u0e00', '‘', '’',
                '\u3000', '。', '上', '东', '京', '份', '会', '公', '北', '双',
                '司', '团', '式', '彤', '技', '料', '新', '方', '日', '有', '术',
                '材', '株', '水', '海', '社', '程', '股', '虹', '防', '限', '集',
                '雨', '﹠', '＆', '（', '）', '，',  '̇']
REPLACE_WITH_SPACE = ['-', ':', '"', '(', ')', "'", "'"]
LEGAL_WORDS = ['ооо', 'оао', 'зао', 'лимитед',
               'llc', 'gmbh', 'inc', 'co', 'ltd', 'sa', 'slr',
               'limited', 'llp', 'ltda', 'corp', 'pty']
STOP_WORDS = ['раша', 'групп', 'asia', 'europe', 'de', 'cv', 'international',
              'pvt', 'trading', 'rubber', 'india', 'private', 'logistics',
              'industries', 's', 'shanghai', 'products', 'mexico', 'shenzhen',
              'imp', 'and', 'usa', 'exp', 'dongguan', 'a', 'plastic', 'group',
              'trade', 'industrial', 'corporation', 'rl', 'sports', 'global',
              'of', 'chemical', 'qingdao', 'technology', 'e', 'services',
              'the', 'mfg', 'y', 'america', 'pt', 'bank', 'textile',
              'chemicals', 'sas', 'enterprises', 'plastics', 'al', 'h', 'srl',
              'polymers', 'brasil', 'materials', 'industry', 'guangzhou',
              'comercio', 'mills', 'p', 'general', 'industria', 'tire',
              'shoes', 'gmb', 'solutions', 'c', 'china', 'systems', 'spinning',
              'auto', 'sp', 'freight', 'city', 'canada', 'l', 'engineering',
              'automotive', 'technologies', 'supply', 'new', 'do', 'sac',
              'service', 'equipment', 'parts', 'sociedad', 'to', 'cargo',
              'shipping', 'express', 'electronics', 'as', 'sdn']


def clean_company_name_string(name: str,
                              drop_symbols: List[str] = DROP_SYMBOLS,
                              space_symbols: List[str] = REPLACE_WITH_SPACE,
                              legal_words: List[str] = LEGAL_WORDS,
                              stop_words: List[str] = STOP_WORDS) -> str:
    ''' Preprocess strings with following steps:
    1. Remove symbols from drop_symbols list
    2. Replace symbols from space_symbols with spaces
    3. Replace symbols with accents to symbols without accents
    4. Remove stop words
    5. Transliterate from russian language (if needed)
    '''
    characters = []
    for c in name.lower():
        if c in drop_symbols:
            continue
        elif c in space_symbols:
            characters.append(' ')
        else:
            characters.append(c)
    # replace symbols with accents
    replace = {'á': 'a',
               'ã': 'a',
               'ç': 'c',
               'è': 'e',
               'é': 'e',
               'í': 'i',
               'ñ': 'n',
               'ó': 'o',
               'õ': 'o',
               'ö': 'o',
               'ú': 'u',
               'ü': 'u',
               'ę': 'e',
               'ł': 'l',
               'ő': 'o',
               'ş': 's',
               'ű': 'u'}
    cleaned_name = ''.join((replace.get(c, c) for c in characters))
    cleaned_name = cleaned_name.strip()
    # remove stop words
    words = cleaned_name.split()
    cleaned_words_legal = [w for w in words if w not in legal_words]
    if cleaned_words_legal == []:
        cleaned_words_legal = words
    cleaned_words = [w for w in cleaned_words_legal if w not in stop_words]
    if cleaned_words == []:
        cleaned_words = cleaned_words_legal
    assembled_name = ' '.join(cleaned_words)
    # transliterate from russian language
    need_ru_transliteration = False
    for c in assembled_name:
        if c >= 'а' and c <= 'я':
            need_ru_transliteration = True
            break
    if need_ru_transliteration:
        return translit(assembled_name, 'ru', reversed=True)
    return assembled_name
