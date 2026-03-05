import numpy as np
import pandas as pd
from typing import Optional, List
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler

class BaseDataPreprocessor(TransformerMixin):
    def __init__(self, needed_columns: Optional[List[str]]=None):
        """
        :param needed_columns: if not None select these columns from the dataframe
        """
        self.scaler = StandardScaler()
        self.columns = needed_columns

    def _select_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.columns is None:
            return data
        return data.filter(items=self.columns)

    def fit(self, data, *args):
        """
        Prepares the class for future transformations
        :param data: pd.DataFrame with all available columns
        :return: self
        """
        data_subset = self._select_columns(data)
        self.scaler.fit(data_subset, args)
        return self

    def transform(self, data: pd.DataFrame) -> np.array:
        """
        Transforms features so that they can be fed into the regressors
        :param data: pd.DataFrame with all available columns
        :return: np.array with preprocessed features
        """
        data_subset = self._select_columns(data)
        return self.scaler.transform(data_subset)