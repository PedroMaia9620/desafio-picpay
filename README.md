# Desafio PicPay

O Documento a seguir descreverá o passo a passo de como rodar o case proposto, para chegar que os avaliadores possam chegar no mesmo resultado. Dado o contexto em questão, irei tomar como presmissa que o case se baseia em desenvolver um modelo de Score de Crédito, onde o target é um indicador de inadimplência. Essas duas premissas são muito importantes para ajudar a nortear em como trabalhar com o modelo e como avaliá-lo posteriormente. 

Nas próximas sessões descrevemos como configurar o ambiente para o projeto também o passo a passo do desenvolvimento.

## Configuração inicial

Para o projeto em questão foi montado um ambiente virtual. Essa etapa foi necessária para que o avaliador tenha o mesmo ambiente que o projeto foi desenvolvido. Para configurar o ambiente basta seguir o passo a passo a seguir:
1. Instale o virtualenv, caso não tenha:
   pip install virtualenv

2. Crie e ative o ambiente virtual:
   virtualenv venv
   source venv/bin/activate  # (macOS/Linux)
   venv\Scripts\activate     # (Windows)

3. Instale as dependências:
   pip install -r requirements.txt

## Estrutura do Projeto

O case foi estruturado pensando em um repositório de projetos reais. A pasta `data` é onde estão todos os arquivos utilizados e criados para o case. Dentro delas há a pasta `raw`, que contém a base `base_modelo.csv` inicial, a pasta `interim` que contém as bases de treino e teste pós feature_engineering e a pasta `processed` que possui os arquivos pós treinamento dos modelos que foram experimentados para o problema em questão. 

A pasta `notebooks` contém todos os arquivos que foram utilizados para explorar a base, explorar possibilidades no feature engineering e na medição do impacto do modelo. Aqui temos 3 subpastas. 
A pasta `eda` possui o jupyter notebook `01_eda.ipynb` que detalha a análise exploratória da `base_modelo.csv`. Esse notebook foi o responsável pelo feature selection e definir as estratégias do treino e teste. 
A pasta `feature_engineering` possui o arquivo `02_feature_eng.ipynb`, o qual mostra a análise feita para a criação de novas variáveis e onde foi possível se apronfundar mais na relação das variáveis selecionadas com o target e também na estabilidade delas ao longo do tempo.
A pasta `model_evaluation` possui 2 notebooks. O primeiro é o arquivo `03_definicao_mvp.ipynb` onde foram testados diversos algoritmos de classificação para definir quais seriam os melhores candidatos para o problema em questão. Por fim o último notebook é o `04_model_performance.ipynb`, onde foi feita a análise do modelo final, porém com uma visão voltada ao negócio. 

A próxima pasta é a `src` onde os scripts em python que utilizamos para desenvolver o modelo de crédito. A pasta `feature_engineering` possuem 6 arquivos script.py, sendo 4 scripts transformers criados para montar um pipeline que foi utilizado no arquivo `feat_eng_pipeline.py`. O pipeline foi criado com o intuito de simular um pipeline de dados em produção e foi utilizado no arquivo `feature_engineering.py`, onde realizamos a filtragem e tratamento do dataset bruto, criamos novas variáveis e por fim separamos a base de treino e teste. 
A última dentro de `src` é a `model`. Essa pasta possui o script `training_model.py` onde definimos um algoritmo para treinar com os dados de treino e avaliamos com o treino e teste. A resposta desse script está salva nas pastas `models` e `outputs`.

Quase todas as pastas e suas etapas possuem um arquivo `utils.py` que contém as funções criadas para auxiliar na tarefa e etapa proposta.

### EDA

A EDA é iniciada no arquivo `01_eda.ipynb` na pasta `notebooks/eda`. Para termos uma EDA clara foram levantadas as seguintes etapas

1. Taxa de Eventos por safra
2. Quantidade de nulos por variáveis
3. Entender qual variável pode ser tratada como categórica e qual pode ser tratada como numérica
4. Quais variáveis são muito correlacionadas em si
5. Análise do Target por variávei
6. Distribuição das Variáveis 

Devido a essas etapas foi possível chegar nas seguintes conclusões:

* Menos da metade das features poderá ser utilizada devido a uma alta frequência de nulos. Foi feita uma filtragem das colunas com até 20% de valores nulos. Esse valor foi escolhido, pois a partir dai, os valores imputados para cobrir os nulos podem começar a atrapalhar no treinamento do modelo. Por se tratar de um Score de Crédito;
* Há algumas features que, apesar de estarem como númericas, possuem comportamento de variáveis categóricas, com valores bem definidos se repetindo entre as safras;
* Há variáveis com correlações muito fortes entre si. Será feito, posteriormente, uma análise de Weight of Evidence para entender quais dessas features as mais fortes e assim eliminar aquelas muito correlacionadas, porém preditoras fracas;
* Muitas variáveis numéricas possuem uma cauda muito extensa para a direita. Para um modelo com comportamento linear isso pode ser um problema, sendo assim testaremos modelos que sejam baseados em árvores. Além disso é visível a necessidade de tratar os outliers nessas features;
* Conseguimos identificar variáveis com um bom ordenamento para o target proposto. Como forma de conseguirmos features estáveis e identificar quais as melhores variáveis preditoras, iremos realizar uma análise de Weight of Evidence, que irá complementar a EDA e já servir como Feature Engineering.

### Feature Engineering

Nessa etapa optamos foi feito uma análise de Peso de Evidência para as features selecionadas na EDA. Isso foi feito para identificar as variáveis que pudessem ser boas preditoras, criar novas features baseadas no WoE e também identificar quais variáveis são estáveis tanto na originação quanto na relação com o target. A análise pode ser encontrada no notebook `02_feature_eng.ipynb` na pasta `notebooks/feature_engineering`. 
Após a definição das estratégias a seguir com a base bruta, foram criados os seguintes transformers:

Nome | Descrição |
--- |------------|                   
`fill_na_transformer.py` | Script com um transformer que preencherá valores vazios de uma lista de features |
`outliers_transformer.py` | Script com um transformer que definirá ranges para uma lista de variáveis |
`pandas_cut_transformer.py` | Script que aplicará um pandas.cut em uma lista de variáveis |
`map_transformer.py` | Script que aplicará um map() em uma lista de variáveis |

Todos os transformers criados foram utilizados no pipeline presente no script `feat_eng_pipeline.py`. Esse tipo de abordagem foi feita para reproduzir a transformação dos dados brutos de um modelo em produção, sendo assim o treinamento feito nas próximas etapas iriam ser mais condizentes com as predições de um modelo implantado. O pipeline gerado foi utilizado no script `feature_engineering.py`, o qual puxa os dados brutos da pasta `data/raw`, aplica o pipeline descrito acima e salva as bases de treino e teste na pasta `data/interim`. 
O script pode ser rodado da seguinte forma:
```
python src/feature_engineering/feature_engineering.py -r base_modelo
```

### Modeling

Durante a etapa de modelagem foi feita uma análise para entender qual classificador iriámos utilizar para treinar o modelo de score de crédito. Foram escolhidos os algoritmos de Regressão Logística, Árvore de Decisão, Random Forest, Gradient Boosting, LightGBM e Explainable Boosting Machine. Com os dados processados pelo script `featue_engineering.py` os algoritmos em questão foram treinados e avaliados no notebook `03_definicao_mvp.ipynb` na pasta `notebooks/model_evaluation`. 
Nessa etapa os modelos tiveram a seguinte performande: 

Modelo | ROC Treino | ROC Teste | KS Treino | KS Teste | Acurácia Treino | Acurácia Teste |
-------|------|------|------|------| ------| ------|                
Logistic Regression | 68.82% | 70.12% | 27.95% | 31.66% | 72.32% | 68.94% |
Decision Tree | 78.03% | 63.93% | 40.82% | 22.19% | 74.63% | 67.67% |
Random Forest | 77.86% | 64.94% | 40.71% | 22.96% | 74.63% | 67.63% |
Gradient Boosting | 70.54% | 69.79% | 30.23% | 30.34% | 72.72% | 68.94% |
LightGBM | 73.76% | 68.44% | 34.85% | 27.40% | 72.91% | 69.05% |
Explainable Boosting Machine | 69.30% | 70.36% | 28.30% | 31.95% | 72.37% | 72.37% |

Analisando a tabela acima, foram selecionados os algoritmos de Gradient Boosting, LightGBM e EBM para seguir para a próxima etapa

O script `training_model.py` irá realizar a tunagem de parâmetros e plotar os gráficos ROC e AUC para o algoritmo selecionado. O script em questão pode ser executado no terminal da seguinte forma: 
```
python src/model/training_model.py -a <algoritmo>
```

Para o script ser executado da melhor maneira ele deve apenas receber os argumentos "gboost", "lgbm" e "ebm".

### Avaliação 
A avaliação do projeto foi feita em 2 etapas: uma pela performance do KS e ROC gerada pelo script `training_model.py` e a segunda parte analisando a performance do ponto de vista de negócios com o modelo escolhido. 
A etapa anterior evidenciou o modelo da Explainable Boosting Machine como o mais apropriado para o problema em questão. Podemos comparar as performances na tabela abaixo

Modelo | ROC Treino | ROC Teste | KS Teste |
-------|------|------|------|------|                  
Gradient Boosting | ~70% | ~70% | 29.93% |
LightGBM | ~70% | ~70% | 31.04% |
Explainable Boosting Machine | ~69% | ~70% | 31.66% |

O algoritmo selecionado tem diversas vantagens, além da performance. Ele torna o modelo mais interpretável, o que pode ser bom para os stakeholders do projeto. 
Com o modelo selecionado foi feita análise das métricas do negócio no notebook `04_model_performance.ipynb` na pasta `notebooks/model_evaluation`. O notebook tem os seguintes passos:

1. Análise de Feature Importance com a biblioteca Interpret
2. Normalizar os dados para gerar um Score
3. Análise de Bad Rate por decil
4. Análise de Bad Rate Acumulada por Decil
5. Criação de Segmentos baseado no Score 

Seguindo esses passos é possível ter a seguinte conclusão sobre o projeto:

* O modelo entregue possui uma performance que está dentro da performance esperada para modelos que tentam predizer a inadimplência;
* O modelo se mostrou eficiente em direcionar com qualidade uma boa análise de crédito da base, podendo servir de apoio para a criação de políticas de preço e de crédito. 
* Importante ressaltar que há um espaço bom de melhorias para o modelo. Infelizmente não tivemos acesso a bases de bureaus externas que em versões futuras podem servir para torná-lo cada vez mais robusto