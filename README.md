# 🧠 Projeto: Reconhecimento Inteligente de Dígitos (MNIST)

▶️ **[Assista ao Vídeo de Apresentação do Projeto](https://drive.google.com/file/d/1cRvCuxDOxECt1qTX4kpEN2oA3nawhCTX/view?usp=sharing)**


## 🎯 Qual o problema o software resolve?
Este sistema foi desenvolvido para resolver o problema de classificação de dados não estruturados (imagens). Ele converte matrizes de pixels brutas extraídas de rascunhos manuscritos em predições digitais assertivas, com o objetivo de comparar a performance matemática e computacional entre algoritmos estatísticos clássicos e Redes Neurais Artificiais.

Além da predição padronizada, o software lida com o fenômeno da **Falsa Certeza (Overconfidence)** em Inteligência Artificial, diagnosticando e provando visualmente as vulnerabilidades de um modelo quando ele é forçado a classificar elementos desconhecidos que fogem à sua distribuição de treino (*Out-of-Distribution*).

## 🛠️ Técnicas e Tecnologias Utilizadas
- **Linguagem Principal:** Python 3
- **Pré-processamento (EDA & Pipeline Vetorial):** `NumPy` e `OpenCV` (Bounding-box cropping, extração Alpha-Channel, Binarização via Otsu e Achatamento de Matrizes para 784 *features*).
- **Modelos de Machine Learning (via `Scikit-Learn`):** 
  - *KNN* (K-Nearest Neighbors) - Geométrico Clássico
  - *Gradient Boosting* - Ensemble
  - *MLPClassifier* - Rede Neural Artificial Densa
- **Avaliação de Diagnóstico:** `Matplotlib`, `Seaborn` (Heatmaps) e `Pandas` (Tabelas de Métricas Ponderadas).
- **Interface Gráfica e Serialização:** `Gradio` (Web App Interativo) e `Joblib` (Armazenamento de Pesos).

## 🚀 Como Executar o Sistema

O sistema foi arquitetado para contornar gargalos clássicos do `tensorflow-macos` e OpenMP em processadores ARM (Apple M1/M2), rodando fluidamente de forma nativa em qualquer máquina graças ao uso exclusivo do robusto ecossistema *Scikit-Learn*.

**1. Instalar as Dependências:**
Abra o terminal na raiz do projeto e instale as bibliotecas requeridas:
```bash
pip install -r requirements.txt
```

**2. Treinar os Modelos (Pipeline de Backend):**
O script central executa todo o pipeline de dados (importação, split estratificado, MinMaxScaler e treinamento) e exporta os cérebros (em formato `.pkl`) para a pasta `models_saved/`.
```bash
python3 src/train_and_save.py
```

**3. Iniciar a Interface Interativa (Frontend):**
Suba a aplicação Web para desenhar seus próprios números no quadro branco.
```bash
python3 app/app.py
```
Acesse no seu navegador pelo link local: `http://127.0.0.1:7860`.

**4. Acessar o Relatório Oficial (Jupyter):**
Abra o arquivo `notebooks/01_relatorio_executivo.ipynb` usando a extensão Jupyter (no VSCode, por exemplo) para percorrer as Fases 1 a 5 da documentação do projeto, incluindo os códigos de gráficos OOD e Tabelas Comparativas de Matrizes de Confusão.

## 💡 Melhorias Futuras (Roadmap)
Embora a aplicação atual garanta predições rápidas (em milissegundos) com uma precisão altíssima (97%), listamos abaixo melhorias estruturais escaláveis:
1. **Redes Neurais Convolucionais (CNN):** Migrar da MLP para uma CNN (via PyTorch) especializada em extração espacial, mitigando a necessidade de centralizar obrigatoriamente o desenho no centro do quadro.
2. **Data Augmentation:** Injetar rotações, ruídos e *zoom* nas 70.000 imagens de treino para preparar a rede neural para caligrafias distorcidas de cenários reais.
3. **Thresholding de Confiança:** Implementar uma trava de segurança (*software gate*) na inferência que rejeite imediatamente análises cuja probabilidade máxima (*predict_proba*) seja inferior a 85%, protegendo o sistema contra Falsa Certeza OOD em produção.
