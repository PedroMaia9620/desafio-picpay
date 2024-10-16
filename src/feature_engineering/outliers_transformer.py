from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd
import numpy as np

class OutlierTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, features, limits_list):
        self.features = features
        self.limits_list = limits_list

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        for i, feature in enumerate(self.features):
            lower, upper = self.limits_list[i]
            X_copy[feature] = X_copy[feature].clip(lower=lower, upper=upper)
        return X_copy
