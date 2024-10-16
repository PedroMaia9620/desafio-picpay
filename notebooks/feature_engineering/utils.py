import pandas as pd
import numpy as np

def woe_table_cat(
        abt: pd.DataFrame,
        target: str,
        feature,
) -> pd.DataFrame:
    """
    `woe_table` calculates a table for WOE analysis
    """

    labels = list(abt[feature].unique())
    groups = [abt[feature].eq(i) for i in labels]

    grouped = pd.Series(
        np.select(groups, labels),
        index=abt.index,
        name="group"
    )
    df = pd.concat([grouped, abt[target]], axis=1)

    output = pd.concat([
        grouped.value_counts().rename("count"),
        grouped.value_counts(normalize=True).rename("count_pct"),
        df.groupby("group")[target].sum().rename("bads"),
    ], axis=1).assign(
        goods=lambda df: df["count"].sub(df["bads"]),
        bads_pct=lambda df: df["bads"].div(df["bads"].sum()),
        goods_pct=lambda df: df["goods"].div(df["goods"].sum()),
        woe=lambda df: np.log(df["bads_pct"].div(df["goods_pct"])),
        diff=lambda df: df["bads_pct"].sub(df["goods_pct"]),
        iv=lambda df: df["diff"].mul(df["woe"]),
        bad_rate=lambda df: df["bads"].div(df["count"]),
        bad_rate_norm_avg=lambda df: df["bad_rate"].div(abt[target].mean())
    )
    return output.sort_values(by='woe').style.format({
        "count": "{:,.0f}",
        "count_pct": "{:.2%}",
        "bads": "{:,.0f}",
        "goods": "{:,.0f}",
        "bads_pct": "{:.0%}",
        "goods_pct": "{:.0%}",
        "woe": "{:.0%}",
        "diff": "{:.0%}",
        "iv": "{:.1%}",
        "bad_rate": "{:.2%}",
        "bad_rate_norm_avg": "{:.0%}"
    })

# Gerando a função para a análise de WoE de variáveis numéricas
# A função funcionará de forma similar à anterior, porém irá receber uma lista de intervalos
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