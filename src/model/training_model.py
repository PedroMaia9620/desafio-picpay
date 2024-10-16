import pandas as pd
import numpy as np 
import shap
import pickle
import logging
import argparse
import logging
from datetime import datetime
from utils import *

FEATURES = ['faixa_VAR_6', 'faixa_VAR_9', 'faixa_VAR_19', 'faixa_VAR_57', 'faixa_VAR_25', 'faixa_VAR_32', 'faixa_VAR_40', 'faixa_VAR_60', 'grupo_VAR_20']
TARGET = 'y'

def save_model(algorithm, model):
    model_filename = f'{algorithm}_model.pkl'
    local_model_path = f"src/model/models/{model_filename}"
    with open(local_model_path, 'wb') as local_model_file:
        pickle.dump(model, local_model_file)
    return local_model_path

def train_eval_model(algorithm):

    treino = pd.read_parquet('data/interim/treino.parquet')
    teste = pd.read_parquet('data/interim/teste.parquet')

    treino = treino.set_index(['id', 'mes_originacao'])
    teste = teste.set_index(['id', 'mes_originacao'])

    X_train = treino[FEATURES]
    y_train = treino[TARGET]

    X_test = teste[FEATURES]
    y_test = teste[TARGET]

    if algorithm == 'gboost':
        model = train_gboost_model(X_train, y_train)

        # Plotando o SHAP para mostrar a ordem de importância das variáveis no modelo
        shap_values = shap.TreeExplainer(model=model).shap_values(X_train)

        empty_list = [""] * X_train.shape[1]

        fig = plt.figure(figsize=(15, 30))
        fig.suptitle(f'SHAP {algorithm}', fontsize=13)
        
        ax1 = fig.add_subplot(221)
        shap.summary_plot(shap_values, features=X_train, show=False, max_display=30, plot_type='bar')
        ax1.tick_params(axis="x", labelsize=10)
        ax1.tick_params(axis="y", labelsize=10)
        ax1.set_xlabel("")

        ax2 = fig.add_subplot(222)
        shap.summary_plot(shap_values,
                            features=X_train,
                            feature_names=empty_list, plot_type='dot', show=False, max_display=30)
        ax2.set_xlabel("")
        plt.savefig(f'src/model/outputs/shap_{algorithm}.png')

    elif algorithm == 'lgbm':
        model = train_lgbm_model(X_train, y_train)

        # Plotando o SHAP para mostrar a ordem de importância das variáveis no modelo
        shap_values = shap.TreeExplainer(model=model).shap_values(X_train)

        empty_list = [""] * X_train.shape[1]

        fig = plt.figure(figsize=(15, 30))
        fig.suptitle(f'SHAP {algorithm}', fontsize=13)
        
        ax1 = fig.add_subplot(221)
        shap.summary_plot(shap_values, features=X_train, show=False, max_display=30, plot_type='bar')
        ax1.tick_params(axis="x", labelsize=10)
        ax1.tick_params(axis="y", labelsize=10)
        ax1.set_xlabel("")

        ax2 = fig.add_subplot(222)
        shap.summary_plot(shap_values,
                            features=X_train,
                            feature_names=empty_list, plot_type='dot', show=False, max_display=30)
        ax2.set_xlabel("")
        plt.savefig(f'src/model/outputs/shap_{algorithm}.png')

    else:
        model = train_ebm_model(X_train, y_train)
        ebm_global = model.explain_global()
        fig = ebm_global.visualize()
        image_path = f"src/model/outputs/importance_{algorithm}.png"
        fig.write_image(image_path)

    # save model
    saved_model = save_model(algorithm, model)
    # mlflow.log_artifact(saved_model)

    y_pred_prob_train = model.predict_proba(X_train)[:, 1]
    y_pred_prob_test = model.predict_proba(X_test)[:, 1]

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Curva ROC-AUC para treino
    fpr_train, tpr_train, _ = roc_curve(y_train, y_pred_prob_train)
    auc_train = roc_auc_score(y_train, y_pred_prob_train)
    
    # Curva ROC-AUC para teste
    fpr_test, tpr_test, _ = roc_curve(y_test, y_pred_prob_test)
    auc_test = roc_auc_score(y_test, y_pred_prob_test)
    
    # Plot ROC-AUC para treino e teste
    ax[0].plot(fpr_train, tpr_train, label=f'Treino (AUC = {auc_train:.2f})')
    ax[0].plot(fpr_test, tpr_test, label=f'Teste (AUC = {auc_test:.2f})')
    ax[0].plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax[0].set_title('Curva ROC-AUC')
    ax[0].set_xlabel('Taxa de Falsos Positivos (FPR)')
    ax[0].set_ylabel('Taxa de Verdadeiros Positivos (TPR)')
    ax[0].legend(loc='lower right')
    
    pos_data = y_pred_prob_test[y_test == 1]
    neg_data = y_pred_prob_test[y_test == 0]

    # Define title metrics
    ks_res = ks_2samp(pos_data, neg_data)
    ks = round(100.0 * ks_res.statistic, 2)
    p_value = round(ks_res.pvalue, 7)

    bins = 1000

    # Define curve
    th = np.linspace(0, 1, bins)
    pos = np.array([np.mean(pos_data <= t) for t in th])
    neg = np.array([np.mean(neg_data <= t) for t in th])
    xmax = abs(neg - pos).argmax()

    ax[1].plot(th, pos, "red", label="1")
    ax[1].plot(th, neg, "blue", label="0")
    ax[1].plot((th[xmax], th[xmax]), (pos[xmax], neg[xmax]), "ks--")
    ax[1].legend(loc="upper left")
    ax[1].set_xlabel("Predicted Probability", fontsize=14)
    ax[1].set_title(f"Kolmogorov–Smirnov", weight="bold", fontsize=16)
    ax[1].text(0.5, 0.1, f"KS={ks}%", fontsize=16)
    ax[1].text(0.5, 0.03, f"p-value={p_value}", fontsize=16)
    ax[1].set_ylabel("Cumulative Probability", fontsize=14)

    fig.savefig(f'src/model/outputs/performance_{algorithm}.png')

    X_train[f'pred_proba_{algorithm}'] = y_pred_prob_train
    X_test[f'pred_proba_{algorithm}'] = y_pred_prob_test

    X_train.to_parquet(f'data/processed/treino_processado_{algorithm}.parquet')
    X_test.to_parquet(f'data/processed/teste_processado_{algorithm}.parquet')

    return model


# Bloco principal de execução do script
if __name__ == '__main__':
    # Captura do tempo de início da execução
    time_start = datetime.now().strftime("%H:%M:%S")

    # Análise dos argumentos da linha de comando
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--algorithm', default='ebm')
    args = parser.parse_args()

    # Chamada da função para treinar e avaliar o modelo LightGBM
    train_eval_model(args.algorithm)

    # Exibição do tempo de início e término da execução
    print("\nHora de início:", time_start,
          "Hora de término:", datetime.now().strftime("%H:%M:%S"))