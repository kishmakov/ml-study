import numpy as np
import pandas as pd
from typing import Optional, List

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import TransformerMixin


interesting_columns = ["Overall_Qual", "Garage_Qual", "Sale_Condition", "MS_Zoning"]
continuous_columns =  ['Lot_Frontage',
 'Lot_Area',
 'Year_Built',
 'Year_Remod_Add',
 'Mas_Vnr_Area',
 'BsmtFin_SF_1',
 'BsmtFin_SF_2',
 'Bsmt_Unf_SF',
 'Total_Bsmt_SF',
 'First_Flr_SF',
 'Second_Flr_SF',
 'Low_Qual_Fin_SF',
 'Gr_Liv_Area',
 'Bsmt_Full_Bath',
 'Bsmt_Half_Bath',
 'Full_Bath',
 'Half_Bath',
 'Bedroom_AbvGr',
 'Kitchen_AbvGr',
 'TotRms_AbvGrd',
 'Fireplaces',
 'Garage_Cars',
 'Garage_Area',
 'Wood_Deck_SF',
 'Open_Porch_SF',
 'Enclosed_Porch',
 'Three_season_porch',
 'Screen_Porch',
 'Pool_Area',
 'Misc_Val',
 'Mo_Sold',
 'Year_Sold',
 'Longitude',
 'Latitude']


class BaseDataPreprocessor(TransformerMixin):
    def __init__(self, needed_columns: Optional[List[str]]=continuous_columns):
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


class OneHotPreprocessor(BaseDataPreprocessor):
    def __init__(self, **kwargs):
        super(OneHotPreprocessor, self).__init__(**kwargs)
        self.oh_columns = interesting_columns
        self.oh_encoder = OneHotEncoder(handle_unknown="ignore")

    def _select_oh_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.filter(items=self.oh_columns)

    def fit(self, data, *args):
        super().fit(data, *args)
        self.oh_encoder.fit(self._select_oh_columns())
        return self


    def transform(self, data):
        data1 = super().transform(data)
        data2 = self.oh_encoder.transform(self._select_oh_columns(data)).toarray()
        return np.hstack([data1, data2])

def make_ultimate_pipeline():
    return Pipeline([
        ("preprocessor", OneHotPreprocessor()),
        ("regressor", Ridge(alpha=10)),
    ])