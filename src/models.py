"""
Fase 3: Implementação e Treinamento dos 3 Modelos
(KNN, XGBoost, MLP com Keras)
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

def build_knn(n_neighbors=5, weights='distance'):
    """Configura e retorna um modelo KNN."""
    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
    return model

def build_gradient_boosting(n_estimators=100, max_depth=3, learning_rate=0.1):
    """Configura e retorna um modelo Gradient Boosting nativo do sklearn (para evitar o crash do XGBoost no Mac)."""
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42
    )
    return model

def build_mlp(hidden_layer_sizes=(128, 64)):
    """
    Constrói um modelo Perceptron Multicamadas (MLP) usando Sklearn
    (para evitar os crashes severos do TensorFlow no Mac M1/M2).
    """
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation='relu',
        solver='adam',
        max_iter=20, # Limitado para treinar rápido localmente
        random_state=42,
        verbose=True
    )
    return model

def train_model(model, X_train, y_train, **kwargs):
    """Função genérica para treinar o modelo."""
    # Como agora usamos Sklearn para todos, o .fit() é igual para os três!
    model.fit(X_train, y_train)
    return model, None
