import logging
import pandas as pd 
import numpy as np 
import argparse
from sklearn.pipeline import Pipeline
from pandas_cut_transformer import CutTransformer
from fill_na_transformer import FillnaTransformer
from outliers_transformer import OutlierTransformer
from map_transformer import MapTransformer
from feat_eng_pipeline import pipeline_transformer

logger = logging.getLogger(__name__)


def _generate_temporal_split_df(df, temporal_threshold, temporal_column):
    df[temporal_column] = pd.to_datetime(df[temporal_column])
    df_train = df[df[temporal_column] <= temporal_threshold]
    df_test = df[df[temporal_column] > temporal_threshold]
    return df_train, df_test


def main(raw_dataset, temporal_threshold, temporal_column):
    
    cols_to_use = ['id', 'safra', 'y', 'VAR_6', 'VAR_9', 'VAR_19', 'VAR_57', 'VAR_20', 'VAR_25', 'VAR_32', 'VAR_40', 'VAR_60']
    input_path = f'data/raw/{raw_dataset}.csv'
    output_path = 'data/interim/'

    df = pd.read_csv(input_path, usecols=cols_to_use)

    # Criando a coluna de mes de originação
    df['mes_originacao'] = pd.to_datetime(df['safra'].astype(str).str[0:4] + '-' + df['safra'].astype(str).str[4:])
    df_pross = pipeline_transformer().transform(df)

    df_train, df_test = _generate_temporal_split_df(df_pross, temporal_threshold, temporal_column)

    df_train.to_parquet(output_path+'treino.parquet')
    df_test.to_parquet(output_path+'teste.parquet')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Realiznado o Feature Engineering'
    )

    parser.add_argument('-r', '--raw_dataset', help='raw dataset filename')
    parser.add_argument('-tt', '--temporal_threshold', default='2014-09-30')
    parser.add_argument('-tc', '--temporal_column', default='mes_originacao')
    
    args = parser.parse_args()
    main(args.raw_dataset, args.temporal_threshold, args.temporal_column)