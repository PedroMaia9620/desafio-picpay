import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
import numpy as np
from sklearn import metrics as skmetrics
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
from interpret.glassbox import ExplainableBoostingClassifier


def train_gboost_model(X, y):
    gboost_estimator = GradientBoostingClassifier(
        random_state=42,
    )
    param_gboost = {    
        'n_estimators': [100, 300, 500],
        'learning_rate':  [0.001, 0.01, 0.1],
        'max_depth': [4, 6, 9],
        'min_samples_split': [2, 5, 10],      
        'max_features': ['sqrt', None]         
    }

    gboost_search = HalvingGridSearchCV(
        gboost_estimator,
        param_gboost,
        scoring='roc_auc',
        n_jobs=-1,
        cv=2,
        verbose=10
    )
    gboost_model = gboost_search.fit(X, y)

    return gboost_model.best_estimator_

def train_lgbm_model(X, y):

    lgb_estimator = LGBMClassifier(
        boosting_type='gbdt',
        objective='binary',
        class_weight='balanced',
        random_state=42,
        verbose=-1
    )

    param_lgbm = {'max_depth': [4, 6, 9],
                    'num_leaves': [4, 6, 8, 12],
                    'learning_rate': [0.001, 0.01, 0.1],
                    'n_estimators': [100, 300, 500]}

    lgbm_search = HalvingGridSearchCV(
        lgb_estimator,
        param_lgbm,
        scoring='roc_auc',
        n_jobs=-1,
        cv=2,
        verbose=10
    )

    lgb_model = lgbm_search.fit(X, y)
    
    return lgb_model.best_estimator_

def train_ebm_model(X, y):
    
    ebm_estimator = ExplainableBoostingClassifier(
        random_state=42,
        interactions=0,
        n_jobs=-1
    )

    param_ebm = {'validation_size':  [0.1, 0.2],
                    'learning_rate': [0.001, 0.01, 0.1],
                    'greedy_ratio': [1.0, 1.5, 2.0],
                    'cyclic_progress':  [0.0, 0.5, 1.0],
                    'smoothing_rounds': [50, 200, 500]}
    
    ebm_search = HalvingGridSearchCV(
        ebm_estimator,
        param_ebm,
        scoring='roc_auc',
        n_jobs=-1,
        cv=2,
        verbose=10
    )

    ebm_model = ebm_search.fit(X, y)

    return ebm_model.best_estimator_