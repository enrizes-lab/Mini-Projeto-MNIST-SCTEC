"""
Fase 2 e Fase 5.1: Pipeline de Pré-processamento e Divisão dos Dados
"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def split_data(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """
    Divide os dados em Treino, Validação e Teste com estratificação.
    A validação é separada do conjunto de treino.
    """
    # Primeiro, separamos o teste
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    # Ajustar o tamanho da validação em relação ao temp
    val_ratio = val_size / (1.0 - test_size)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def scale_data(X_train, X_val, X_test):
    """
    Aplica MinMaxScaler para redimensionar pixels para [0, 1].
    Retorna X_train_scaled, X_val_scaled, X_test_scaled e o objeto scaler para referência futura.
    """
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def mask_classes_for_ood(X_train, y_train, classes_to_hide=[4, 7]):
    """
    Remove classes específicas da base de treino (Desafio A - Class Masking).
    Isso testa a generalização extrema (OOD) quando o modelo não conhece essas classes no treino.
    Retorna X_train e y_train filtrados.
    """
    # Cria uma máscara booleana onde True significa "manter" (a classe não está na lista oculta)
    mask = ~np.isin(y_train, classes_to_hide)
    
    X_train_filtered = X_train[mask]
    y_train_filtered = y_train[mask]
    
    return X_train_filtered, y_train_filtered
