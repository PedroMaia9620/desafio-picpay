from scipy.stats import ks_2samp
import pandas as pd
import numpy as np 

def calculate_ks(y_true, y_pred_proba):

    # Separar as probabilidades para as classes 0 e 1
    probas_class_0 = y_pred_proba[y_true == 0]
    probas_class_1 = y_pred_proba[y_true == 1]
    
    # Calcular a estatística KS usando o teste ks_2samp
    ks_stat = ks_2samp(probas_class_1, probas_class_0).statistic
    
    return ks_stat

def woe_table(
        grouped: pd.Series,
        target: pd.Series,
    ) -> pd.DataFrame:
    """
    `woe_table` calculates a table for WOE analysis
    """
    
    df = pd.concat([grouped, target], axis=1)
    
    return pd.concat([
        grouped.value_counts().rename("count"),
        grouped.value_counts(normalize=True).rename("count_pct"),
        df.groupby(grouped.name)[target.name].sum().rename("bads"),
    ], axis=1).assign(
        goods=lambda df: df["count"].sub(df["bads"]),
        bads_pct=lambda df: df["bads"].div(df["bads"].sum()),
        goods_pct=lambda df: df["goods"].div(df["goods"].sum()),
        woe=lambda df: np.log(df["bads_pct"].div(df["goods_pct"])),
        diff=lambda df: df["bads_pct"].sub(df["goods_pct"]),
        iv=lambda df: df["diff"].mul(df["woe"]),
        bad_rate=lambda df: df["bads"].div(df["count"]),
        bad_rate_norm_avg=lambda df: df["bad_rate"].div(target.mean())
    )