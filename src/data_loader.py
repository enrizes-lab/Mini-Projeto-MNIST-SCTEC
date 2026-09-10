"""
Fase 1: Carregamento do dataset MNIST
"""
import numpy as np
import urllib.request
import os

def load_mnist_data():
    """
    Baixa o dataset MNIST via URL direta (Google Storage) e carrega via NumPy puro.
    Isso evita os deadlocks do Scikit-Learn e os crashes C++ do TensorFlow no Mac.
    Retorna: X (features achatadas para 784), y (labels)
    """
    print("Baixando o dataset MNIST...")
    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
    filepath = "mnist.npz"
    
    if not os.path.exists(filepath):
        urllib.request.urlretrieve(url, filepath)
        
    with np.load(filepath, allow_pickle=True) as f:
        X_train, y_train = f['x_train'], f['y_train']
        X_test, y_test = f['x_test'], f['y_test']
    
    # Juntar treino e teste para que nossa função split_data cuide da divisão corretamente
    X_full = np.concatenate((X_train, X_test), axis=0)
    y_full = np.concatenate((y_train, y_test), axis=0)
    
    # Achatar as imagens de (28, 28) para (784,)
    X_flattened = X_full.reshape(-1, 784)
    
    print("Finalizou o processo de baixar o dataset MNIST!")
    return X_flattened, y_full


if __name__ == "__main__":
    X, y = load_mnist_data()
    print(f"X shape: {X.shape}, y shape: {y.shape}")
