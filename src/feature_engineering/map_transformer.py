from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd
import numpy as np

class MapTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, features, mapping_dicts):
        self.features = features
        self.mapping_dicts = mapping_dicts

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        for i, feature in enumerate(self.features):
            mapping_dict = self.mapping_dicts[i]
            X_copy[f'grupo_{feature}'] = X_copy[feature].map(mapping_dict)
        return X_copy
