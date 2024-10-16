import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
from galeritas import bar_plot_with_population_proportion

def cramerV(label,x):
    confusion_matrix = pd.crosstab(label, x)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r,k = confusion_matrix.shape
    phi2 = chi2/n
    phi2corr = max(0,phi2-((k-1)*(r-1))/(n-1))
    rcorr = r - ((r - 1) ** 2) / ( n - 1 )
    kcorr = k - ((k - 1) ** 2) / ( n - 1 )
    try:
        if min((kcorr - 1),(rcorr - 1)) == 0:
            warnings.warn(
            "Não foi possível calcular a correlação de Cramer",RuntimeWarning)
            v = 0
        else:
            v = np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))
    except:
        v = 0
    return v

def plot_cramer(df):
    cramer = pd.DataFrame(index=df.columns,columns=df.columns)
    for column_of_interest in df.columns:
        try:
            temp = {}

            columns = df.columns
            for j in range(0,len(columns)):
                v = cramerV(df[column_of_interest],df[columns[j]])
                cramer.loc[column_of_interest,columns[j]] = v
                if (column_of_interest==columns[j]):
                    pass
                else:
                    temp[columns[j]] = v
            cramer.fillna(value=np.nan,inplace=True)
        except:
            pass
    plt.figure(figsize=(7,7))
    sns.heatmap(cramer,annot=True,fmt='.2f')

    plt.title("Valores da Correlação de Cramer")
    plt.show()

def bar_plot_population_numeric(df, feature, target):
    
    print(f'Proporção de Nulos de {feature}: {df[feature].isnull().mean()}')
    df[f'{feature}'].fillna(-2, inplace=True)

    if df[feature].nunique() == 2:
        df[feature] = df[feature].astype(str)
        print(f'Proporção de Nulos de {feature}: {df[feature].isnull().mean()}')
        bar_plot_with_population_proportion(df,x=feature, y=target, func=np.mean, plot_title=f'Média de {target} por {feature}')
    else:
        try:
            bucket = pd.qcut(df[feature], 8, retbins=True)[1]

            df[f'faixa_{feature}'] = pd.cut(df[feature], bucket, labels=[i for i in range (0, len(bucket)-1)])
            df[f'faixa_{feature}'] = df[f'faixa_{feature}'].astype(str)
            df.loc[df[f'faixa_{feature}'].eq('nan'), f'faixa_{feature}'] = '-1'
            bar_plot_with_population_proportion(df[df[f'faixa_{feature}'].ne('nan')],x=f'faixa_{feature}', y=target, func=np.mean, plot_title=f'Média de {target} por faixa_{feature}')
        except: 
            bucket = pd.qcut(df[feature], 8, duplicates='drop', retbins=True)[1]

            df[f'faixa_{feature}'] = pd.cut(df[feature], bucket, labels=[i for i in range (0, len(bucket)-1)])
            df[f'faixa_{feature}'] = df[f'faixa_{feature}'].astype(str)
            df.loc[df[f'faixa_{feature}'].eq('nan'), f'faixa_{feature}'] = '-1'
            bar_plot_with_population_proportion(df[df[f'faixa_{feature}'].ne('nan')],x=f'faixa_{feature}', y=target, func=np.mean, plot_title=f'Média de {target} por faixa_{feature}')


def bar_plot_population_categorical(df, feature, target):
    # Calcular a frequência de cada categoria
    frequencia = df[feature].value_counts(normalize=True)

    # Filtrar as categorias com frequência de pelo menos 5%
    categorias_filtro = frequencia[frequencia >= 0.03].index

    # Filtrar o DataFrame original para incluir apenas essas categorias
    df_filtrado = df[df[feature].isin(categorias_filtro)]
    df_filtrado[feature] = df_filtrado[feature].astype(str)
    print(f'Proporção de Nulos de {feature}: {df_filtrado[feature].isnull().mean()}')
    bar_plot_with_population_proportion(df_filtrado[df_filtrado[feature].ne('nan')],x=feature, y=target, func=np.mean, plot_title=f'Média de {target} por {feature}')