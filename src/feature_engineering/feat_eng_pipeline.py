import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from pandas_cut_transformer import CutTransformer
from fill_na_transformer import FillnaTransformer
from outliers_transformer import OutlierTransformer
from map_transformer import MapTransformer

def pipeline_transformer():
    return Pipeline([
        ('fillna_transformer', FillnaTransformer(
            features=['VAR_6', 'VAR_9', 'VAR_19', 'VAR_57', 'VAR_20', 'VAR_25', 'VAR_32', 'VAR_40', 'VAR_60'],
            fill_values=[-100, 3000.0, 0, 0, 12, 0.0, -1.0, 0.0, -1.0]
        )),
        ('outlisers_transformer', OutlierTransformer(
            features=['VAR_6', 'VAR_9', 'VAR_19'],
            limits_list=[(None, 4959.77), (None, 3000.0), (None, 86.0)]
        )),
        ('cut_transformer', CutTransformer(
            features=['VAR_6',
                      'VAR_9', 
                      'VAR_19', 
                      'VAR_57',
                      'VAR_25',
                      'VAR_32',
                      'VAR_40',
                      'VAR_60'
                      ],
            bins_list=[[-np.inf, 380.0, 679.4, 1166.0, np.inf], 
                       [-np.inf, 424.901, 570.0, 1000.0, np.inf],
                       [-np.inf, 0, 13, np.inf],
                       [-np.inf, 38.0, 49.0, 64.0, np.inf],
                       [-np.inf, 2.0, np.inf],
                       [-np.inf, 0.08, 0.13, np.inf],
                       [-np.inf, 0.0, 8.0, np.inf],
                       [-np.inf, -0.35, 0, 0.4, np.inf]
                       ],
            labels_list=[[1, 2, 3, 4],
                         [4, 3, 2, 1],
                         [1, 2, 3],
                         [1, 2, 3, 4],
                         [1, 2],
                         [1, 2, 3],
                         [1, 2, 3],
                         [1, 2, 3, 4]
                         ]
        )),
        ("map_transformer", MapTransformer(
            features=['VAR_20'],
            mapping_dicts=[{
                3: 4,
                4: 4,
                7: 4,
                5: 3,
                6: 3,
                8: 3,
                9: 2,
                10: 2,
                11: 1, 
                12: 1
            }
            ]
        ))
    ])