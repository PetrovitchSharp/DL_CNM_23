import argparse
import sys

sys.path.append('../')

import pandas as pd
from fuzzywuzzy import fuzz
import Levenshtein
from tqdm import tqdm

from utils.text_preparation import clean_and_concat, transliterate
from utils.feature_extraction import cos_distance, jaccard_similarity


def make_parser() -> argparse.ArgumentParser:
    '''
    Make cli arguments parser

    Returns:
        CLI args parser
    '''
    parser = argparse.ArgumentParser(
        description='Prepare data for catboost model'
    )

    parser.add_argument('-data', type=str,
                        default='train.csv',
                        help='raw dataset filename')

    parser.add_argument('-output', type=str,
                        default='train.csv',
                        help='prepared dataset filename')

    return parser

def main() -> None:
    '''
    Main function responsible for catboost dataset preparation
    '''
    # Arguments parsing
    parser = make_parser()
    args = parser.parse_args()

    data_path = args.data
    output_path = args.output

    print('Preparing dataset...')

    pbar = tqdm(total = 22)

    # Load raw dataset
    data = pd.read_csv(
        f'../../data/raw/{data_path}',
        index_col='pair_id'
    )

    pbar.update(1)

    # Lowercase translation for ease of processing
    data.name_1 = data.name_1\
        .apply(str.lower)
    data.name_2 = data.name_2\
        .apply(str.lower)

    pbar.update(2)

    # Transliteraion
    data['transliterated_name_1'] = data.name_1\
        .apply(transliterate)
    data['transliterated_name_2'] = data.name_2\
        .apply(transliterate)

    pbar.update(2)

    # Cleaning and concatenation
    data['concated_name_1'] = data.transliterated_name_1\
        .apply(clean_and_concat)
    data['concated_name_2'] = data.transliterated_name_2\
        .apply(clean_and_concat)

    pbar.update(2)

    # Ratio of similar characters on both string
    data['trans_wratio'] = data.apply(
        lambda row: fuzz.WRatio(
            row.transliterated_name_1,
            row.transliterated_name_2
        ) / 100, 
        axis = 1
    )

    data['conc_wratio'] = data.apply(
        lambda row: fuzz.WRatio(
            row.concated_name_1,
            row.concated_name_2
        ) / 100, 
        axis = 1
    )

    pbar.update(2)

    # Ratio of the most similar substring
    data['trans_partial_ratio'] = data.apply(
        lambda row: fuzz.partial_ratio(
            row.transliterated_name_1,
            row.transliterated_name_2
        ) / 100, 
        axis = 1
    )

    data['conc_partial_ratio'] = data.apply(
        lambda row: fuzz.partial_ratio(
            row.concated_name_1, 
            row.concated_name_2
        ) / 100,
        axis = 1
    )

    pbar.update(2)

    # Measure of the sequences' tokens similarity
    data['trans_token_sort_ratio'] = data.apply(
        lambda row: fuzz.token_sort_ratio(
            row.transliterated_name_1,
            row.transliterated_name_2
        ) / 100,
        axis = 1
    )

    data['conc_token_sort_ratio'] = data.apply(
        lambda row: fuzz.token_sort_ratio(
            row.concated_name_1, 
            row.concated_name_2
        ) / 100,
        axis = 1
    )

    pbar.update(2)

    # Levenshtein distance
    data['trans_levenshtein'] = data.apply(
        lambda row: Levenshtein.distance(
            row.transliterated_name_1,
            row.transliterated_name_2
        ),
        axis = 1
    )

    data['conc_levenshtein'] = data.apply(
        lambda row: Levenshtein.distance(
            row.concated_name_1, 
            row.concated_name_2
        ), 
        axis = 1
    )

    pbar.update(2)

    # Levenshtein distance normalized to the maximum length
    data['trans_levenshtein_ratio'] = data.apply(
        lambda row: Levenshtein.distance(
            row.transliterated_name_1,
            row.transliterated_name_2
        ) / max(
            len(row.transliterated_name_1),
            len(row.transliterated_name_2)
        ), 
        axis = 1
    )

    data['conc_levenshtein_ratio'] = data.apply(
        lambda row: Levenshtein.distance(
            row.concated_name_1, 
            row.concated_name_2
        ) / max(
            len(row.concated_name_1),
            len(row.concated_name_2)
        ), 
        axis = 1
    )

    pbar.update(2)

    # Jaro distance
    data['trans_jaro'] = data.apply(
        lambda row: Levenshtein.jaro(
            row.transliterated_name_1,
            row.transliterated_name_2
        ),
        axis = 1
    )

    data['conc_jaro'] = data.apply(
        lambda row: Levenshtein.jaro(
            row.transliterated_name_1, 
            row.concated_name_2
        ), 
        axis = 1
    )

    pbar.update(2)

    # Cosine distance between strings
    data['trans_cosine'] = data.apply(
        lambda row: cos_distance(
            row.transliterated_name_1, 
            row.transliterated_name_2
        ), 
        axis = 1
    )

    pbar.update(1)

    # Jaccard similarity of strings
    data['trans_jaccard'] = data.apply(
        lambda row: jaccard_similarity(
            row.transliterated_name_1, 
            row.transliterated_name_2
        ), 
        axis = 1
    )

    pbar.update(1)

    # Dropping all non-numerical columns
    data = data.drop(
        ['name_1',
        'name_2',
        'concated_name_1',
        'concated_name_2',
        'transliterated_name_1',
        'transliterated_name_2'
        ],
        axis=1
    )

    pbar.update(1)
    pbar.close()

    # Saving of preprocessed dataset
    data.to_csv(f'../../data/processed/{output_path}')

    print('Dataset has been prepared')



if __name__ == '__main__':
    main()