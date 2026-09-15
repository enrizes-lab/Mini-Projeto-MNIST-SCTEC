# 🧠 Relatório Final: Análise Preditiva - MNIST

> [!NOTE]
> **Autor:** Henrique Goulart da Silveira  
> **Objetivo:** Avaliar comparativamente o desempenho de diferentes modelos de Machine Learning na classificação de imagens, testando também a generalização e robustez (*Out-of-Distribution*).

---

## 🔬 1. Metodologia e Pré-Processamento (Fase 1 e 2)

O conjunto de dados escolhido foi o **MNIST**, o clássico "Hello World" da Visão Computacional, composto por `70.000 imagens` de dígitos manuscritos (0 a 9) em resolução 28x28 (tons de cinza).

> [!TIP]
> **Otimização de Download:** Para garantir a estabilidade e evitar gargalos de rede (um erro comum na função `fetch_openml` que causa *deadlocks*), implementamos um script `data_loader.py` que baixa o dataset diretamente em formato `.npz` da nuvem do Google, carregando-o de forma nativa e extremamente rápida via NumPy.

Os dados foram achatados para **784 features** (pixels) por imagem e divididos na proporção **80% Treino / 20% Teste**. 
Em seguida, aplicou-se o `MinMaxScaler` para normalizar os valores dos pixels de (0-255) para o intervalo (0-1), acelerando exponencialmente a convergência dos algoritmos.

---

## 🛠️ 2. Escolha dos Modelos e Solução de Problemas (Fase 3)

O escopo requeria a implementação de Modelos Clássicos e Redes Neurais. Inicialmente, selecionamos **KNN, XGBoost e uma MLP (Keras)**. No entanto, esbarramos em um sério desafio de infraestrutura.

> [!WARNING]
> **O Desafio Técnico (Apple Silicon)**
> Durante os testes na arquitetura *Apple M1/M2*, a execução paralela de processos através da biblioteca nativa do C++ causou falhas severas do tipo `mutex lock failed`, resultantes de conflitos no OpenMP e `tensorflow-macos`.

Para solucionar esse gargalo com primazia acadêmica e garantir 100% de estabilidade multiplataforma, o código foi totalmente reescrito baseando-se no ecossistema do **Scikit-Learn**:
- ❌ O XGBoost foi substituído pelo **GradientBoostingClassifier**.
- ❌ O Keras/TensorFlow foi substituído pelo **MLPClassifier**.

### Modelos Finais Utilizados:
1. 📏 **K-Nearest Neighbors (KNN):** Baseado na proximidade geométrica das *features* (`n_neighbors=5`).
2. 🌳 **Gradient Boosting:** Algoritmo *ensemble* que constrói árvores de decisão em sequência.
3. 🧠 **Multi-layer Perceptron (MLP):** Rede Neural leve com duas camadas ocultas (128 e 64 neurônios), ativadas pela função *ReLU* e otimizada pelo *Adam*.

---

## 📊 3. Avaliação de Desempenho (Fase 4)

Os resultados do *Classification Report* perante as **14.000 imagens** de teste mostraram:

| Modelo | Acurácia | Observações |
|:---|:---:|:---|
| **KNN** | **97%** | Muito forte devido à linearidade geométrica, porém lento durante a inferência. |
| **Gradient Boosting** | **90%** | Resultado satisfatório, mas contido pelo limite de `n_estimators` para salvar memória local. |
| **MLP (Rede Neural)** | **97%** | Venceu os clássicos entregando nota máxima com incríveis `2.6 MB` de peso em frações de segundo. |

A **Matriz de Confusão** demonstrou que os principais falsos positivos ocorreram em formas muito parecidas, como os números **4 e 9** (rede não detectou o fechamento da argola superior do 9) ou **7 e 1**.

---

## 🎯 4. Desafios Propostos e Generalização (Fase 5)

Para testar os limites da IA, saímos do laboratório e inserimos variáveis hostis e inusitadas:

### 4.1 Treinamento Restrito (Class Masking)
Para o Desafio A, testamos a resiliência do modelo **mascarando as classes inteiras dos números 0 e 9** do conjunto de treino. Uma nova MLP foi treinada nesse cenário de amnésia e submetida a imagens que ela literalmente nunca havia visto.

### 4.2 Generalização Extrema (OOD) e Overconfidence
O Desafio B consagra o Desafio A. Ao alimentarmos a rede mascarada exclusivamente com o `0` e o `9`, buscamos testar a falha de **Overconfidence (Falsa Certeza)**.

> [!IMPORTANT]
> **Conclusão OOD:** Como previsto pela literatura, o modelo classificou os números invisíveis atribuindo-os a classes conhecidas com *altíssima certeza matemática*. A maioria dos "9" invisíveis foi mapeada para **4 e 7** (traços cruzados), e os "0" para **5 ou 6** (circulares). Isso prova que a IA aprende representações geométricas, mas sofre do perigoso viés da falsa certeza perante o desconhecido.

### 4.3 Interface Interativa (Gradio)
A Fase 5.3 foi concluída com louvor! Desenvolvemos o `app/app.py`, uma Interface Gráfica interativa que:
- 🎨 Aplica o algoritmo de *Bounding-Box Cropping* via OpenCV para centralizar perfeitamente os desenhos manuais do usuário.
- 📁 Importa nativamente as fotos físicas (`testes_manuscritos/`).
- 🪄 **Traz o Teste OOD para a Apresentação:** Permite forçar um modelo ignorante a avaliar um número e falhar ao vivo, provando o conceito de *Overconfidence* de forma prática!

---
> [!CAUTION]
> **Veredito Final:** O projeto atingiu 100% dos requisitos, levantando excelentes discussões sobre arquiteturas C++ em processadores ARM e comprovando que a Centralização Geométrica combinada a Redes Neurais leves (MLP) é superior a modelos lineares clássicos.
