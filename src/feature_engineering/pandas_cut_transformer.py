from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd

class CutTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, features, bins_list, labels_list):
        self.features = features
        self.bins_list = bins_list
        self.labels_list = labels_list

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        for i, feature in enumerate(self.features):
            bins = self.bins_list[i]
            labels = self.labels_list[i]
            X_copy[f'faixa_{feature}'] = pd.cut(X_copy[feature], bins=bins, labels=labels)
        return X_copy
