"""
Fase 4 e Fase 5.2: Avaliação Comparativa de Desempenho e Generalização Extrema
"""
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

import numpy as np

def plot_confusion_matrix(y_true, y_pred, labels=None, title="Matriz de Confusão"):
    """
    Plota um mapa de calor (heatmap) da matriz de confusão.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    
    # Seaborn quebra se passarmos explícitamente xticklabels=None, o padrão dele é 'auto'
    tick_labels = labels if labels is not None else 'auto'
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=tick_labels, yticklabels=tick_labels)
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.title(title)
    plt.show()

def evaluate_model(model, X_test, y_test, is_keras=False):
    """
    Calcula métricas e gera o classification report.
    """
    if is_keras:
        # Keras retorna probabilidades para cada classe
        y_prob = model.predict(X_test)
        y_pred = np.argmax(y_prob, axis=1)
    else:
        # Sklearn/XGBoost retornam a classe predita diretamente
        y_pred = model.predict(X_test)
        
    print("Classification Report:")
    report = classification_report(y_test, y_pred)
    print(report)
    return y_pred, report

def analyze_overconfidence(model, X_ood, y_ood_true, class_names=None, is_keras=False):
    """
    Analisa como o modelo se comporta com dados Fora de Distribuição (OOD).
    Retorna as predições e as probabilidades (predict_proba).
    """
    if is_keras:
        y_prob = model.predict(X_ood)
    else:
        y_prob = model.predict_proba(X_ood)
        
    y_pred = np.argmax(y_prob, axis=1)
    return y_pred, y_prob
