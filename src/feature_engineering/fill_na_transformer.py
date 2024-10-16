from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd
import numpy as np

class FillnaTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, features, fill_values):
        self.features = features
        self.fill_values = fill_values

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        for i, feature in enumerate(self.features):
            X_copy[feature] = X_copy[feature].fillna(self.fill_values[i])
        return X_copy
