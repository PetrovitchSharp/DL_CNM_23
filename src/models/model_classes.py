from typing import Any
from abc import ABC, abstractmethod

import pandas as pd
import Levenshtein
from catboost import CatBoostClassifier
from fuzzywuzzy import fuzz

from utils.text_preparation import transliterate, clean_and_concat
from utils.feature_extraction import cos_distance, jaccard_similarity


class BasicModel(ABC):
    '''
    Basic class for all models
    '''
    @abstractmethod
    def __init__(self, model: Any) -> None:
        pass

    @abstractmethod
    def predict(self, company_name_1: str, company_name_2: str) -> bool:
        '''
        Predicts whether the input company names
        are the name of a single company

        Args:
            company_name_1: First company name
            company_name_2: Second company name

        Returns:
            True if company names represents single company,
            False otherwise
        '''
        pass


class CatboostModel(BasicModel):
    def __init__(self, model: CatBoostClassifier) -> None:
        self.model = model

    def preprocess_input(self, company_name_1: str,
                         company_name_2: str) -> pd.Series:
        '''
        Extract numerical features from input company names

        Args:
            company_name_1: First company name
            company_name_2: Second company name

        Returns:
            Pandas series with numerical features
        '''
        # Transliterate company names
        transliterated_name_1 = transliterate(company_name_1.lower())
        transliterated_name_2 = transliterate(company_name_2.lower())

        # Concat and clear company names from characters
        # that are not numbers or letters
        concated_name_1 = clean_and_concat(company_name_1.lower())
        concated_name_2 = clean_and_concat(company_name_2.lower())

        # Numerical features based on transliterated names
        trans_wratio = fuzz.WRatio(
            transliterated_name_1,
            transliterated_name_2
        ) / 100
        trans_partial_ratio = fuzz.partial_ratio(
            transliterated_name_1,
            transliterated_name_2
        ) / 100
        trans_token_sort_ratio = fuzz.token_sort_ratio(
            transliterated_name_1,
            transliterated_name_2
        ) / 100
        trans_levenshtein = Levenshtein.distance(
            transliterated_name_1,
            transliterated_name_2
        )
        trans_levenshtein_ratio = Levenshtein.distance(
            transliterated_name_1,
            transliterated_name_2
        ) / max(
            len(transliterated_name_1),
            len(transliterated_name_2)
        )
        trans_jaro = Levenshtein.jaro(
            transliterated_name_1,
            transliterated_name_2
        )
        trans_cosine = cos_distance(
            transliterated_name_1,
            transliterated_name_2
        )
        trans_jaccard = jaccard_similarity(
            transliterated_name_1,
            transliterated_name_2
        )

        # Numerical features based on concatenated and cleared names
        conc_wratio = fuzz.WRatio(
            concated_name_1,
            concated_name_2
        ) / 100
        conc_partial_ratio = fuzz.partial_ratio(
            concated_name_1,
            concated_name_2
        ) / 100
        conc_token_sort_ratio = fuzz.token_sort_ratio(
            concated_name_1,
            concated_name_2
        ) / 100
        conc_levenshtein = Levenshtein.distance(
            concated_name_1,
            concated_name_2
        )
        conc_levenshtein_ratio = Levenshtein.distance(
            concated_name_1,
            concated_name_2
        ) / max(
            len(concated_name_1),
            len(concated_name_2)
        )
        conc_jaro = Levenshtein.jaro(
            concated_name_1,
            concated_name_2
        )

        return pd.Series({
            'trans_wratio': trans_wratio,
            'trans_partial_ratio': trans_partial_ratio,
            'trans_token_sort_ratio': trans_token_sort_ratio,
            'trans_levenshtein': trans_levenshtein,
            'trans_levenshtein_ratio': trans_levenshtein_ratio,
            'trans_jaro': trans_jaro,
            'trans_cosine': trans_cosine,
            'trans_jaccard': trans_jaccard,
            'conc_wratio': conc_wratio,
            'conc_partial_ratio': conc_partial_ratio,
            'conc_token_sort_ratio': conc_token_sort_ratio,
            'conc_levenshtein': conc_levenshtein,
            'conc_levenshtein_ratio': conc_levenshtein_ratio,
            'conc_jaro': conc_jaro
        })

    def predict(self, company_name_1: str, company_name_2: str) -> bool:
        input_data = self.preprocess_input(company_name_1, company_name_2)

        return self.model.predict(input_data) == 1
