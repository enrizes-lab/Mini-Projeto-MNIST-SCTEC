import os
# Correção crucial para Macs com chip M1/M2 (Apple Silicon) para evitar crash de C++ no XGBoost/OpenMP
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'

import joblib
import numpy as np
from preprocess import split_data, scale_data
from data_loader import load_mnist_data
from models import build_knn, build_gradient_boosting, build_mlp, train_model

def main():
    print("1. Carregando dados...")
    X, y = load_mnist_data()
    
    # Reduzindo o dataset para treinar mais rápido localmente (opcional, remova na versão final)
    # X, y = X[:20000], y[:20000] 
    
    print("2. Dividindo os dados...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    print("3. Normalizando os dados...")
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_data(X_train, X_val, X_test)
    
    # Criar pasta para salvar modelos
    os.makedirs('models_saved', exist_ok=True)
    joblib.dump(scaler, 'models_saved/scaler.pkl')
    
    # --- KNN ---
    print("\n--- Treinando KNN ---")
    knn = build_knn(n_neighbors=5)
    knn, _ = train_model(knn, X_train_scaled, y_train)
    joblib.dump(knn, 'models_saved/knn.pkl')
    print("KNN salvo!")

    # --- Gradient Boosting (Sklearn) ---
    print("\n--- Treinando Gradient Boosting ---")
    xgb_model = build_gradient_boosting(n_estimators=30, max_depth=3) # Parâmetros reduzidos p/ velocidade
    xgb_model, _ = train_model(xgb_model, X_train_scaled, y_train)
    joblib.dump(xgb_model, 'models_saved/xgb.pkl')
    print("Gradient Boosting salvo!")

    # --- MLP (Sklearn) ---
    print("\n--- Treinando MLP ---")
    mlp = build_mlp(hidden_layer_sizes=(128, 64))
    mlp, _ = train_model(mlp, X_train_scaled, y_train)
    joblib.dump(mlp, 'models_saved/mlp.pkl')
    print("MLP salvo!")
    
    # --- MLP Masked (Para o Desafio OOD) ---
    print("\n--- Treinando MLP Masked (Sem 0 e 9) ---")
    mask_train = ~np.isin(y_train, [0, 9])
    X_train_masked = X_train_scaled[mask_train]
    y_train_masked = y_train[mask_train]
    
    mlp_masked = build_mlp(hidden_layer_sizes=(64,)) # Menor para ser mais rápido
    mlp_masked, _ = train_model(mlp_masked, X_train_masked, y_train_masked)
    joblib.dump(mlp_masked, 'models_saved/mlp_masked.pkl')
    print("MLP Masked salvo!")
    
    print("\nTodos os modelos foram treinados e salvos na pasta 'models_saved'!")

if __name__ == "__main__":
    main()
