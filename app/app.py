"""
Fase 5.3: Interface Gradio Interativa com Desafios Acadêmicos
"""
import gradio as gr
import numpy as np
import joblib
import cv2
import glob

# Tentar carregar os modelos (podem não existir ainda se o script de treino não rodou)
try:
    scaler = joblib.load('models_saved/scaler.pkl')
    knn_model = joblib.load('models_saved/knn.pkl')
    xgb_model = joblib.load('models_saved/xgb.pkl')
    mlp_model = joblib.load('models_saved/mlp.pkl')
    try:
        mlp_masked = joblib.load('models_saved/mlp_masked.pkl')
    except:
        mlp_masked = None
    models_loaded = True
except Exception as e:
    print(f"Erro ao carregar modelos: {e}")
    models_loaded = False

def process_image(image_dict):
    """
    Função utilitária para limpar, centralizar e escalar a imagem.
    """
    if isinstance(image_dict, dict) and 'composite' in image_dict:
        img = image_dict['composite']
    else:
        img = image_dict

    if img is None:
        return None

    if len(img.shape) == 3 and img.shape[2] == 4:
        img = img[:, :, 3]
    elif len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
    if len(img.shape) == 2:
        if np.mean(img[0:5, 0:5]) > 127:
            img = cv2.bitwise_not(img)
            _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    coords = cv2.findNonZero(img)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        img_cropped = img[y:y+h, x:x+w]
        
        size = max(w, h)
        pad_x = (size - w) // 2
        pad_y = (size - h) // 2
        img_padded = cv2.copyMakeBorder(img_cropped, pad_y, size - h - pad_y, pad_x, size - w - pad_x, cv2.BORDER_CONSTANT, value=0)
        
        margin = int(size * 0.15)
        img_padded = cv2.copyMakeBorder(img_padded, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=0)
        img = img_padded

    img_resized = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    img_flattened = img_resized.flatten().reshape(1, -1)
    return scaler.transform(img_flattened)

def predict_digit(image_dict):
    if not models_loaded:
        return {"Modelos não carregados": 1.0}, {"Modelos não carregados": 1.0}, {"Modelos não carregados": 1.0}
    try:
        img_scaled = process_image(image_dict)
        if img_scaled is None:
            return {"Vazio": 1.0}, {"Vazio": 1.0}, {"Vazio": 1.0}
        
        prob_knn = knn_model.predict_proba(img_scaled)[0]
        prob_xgb = xgb_model.predict_proba(img_scaled)[0]
        prob_mlp = mlp_model.predict_proba(img_scaled)[0]
        
        res_knn = {str(i): float(prob_knn[i]) for i in range(10)}
        res_xgb = {str(i): float(prob_xgb[i]) for i in range(10)}
        res_mlp = {str(i): float(prob_mlp[i]) for i in range(10)}
        return res_knn, res_xgb, res_mlp
    except Exception as e:
        return {"Erro na Imagem": 1.0}, {"Erro na Imagem": 1.0}, {"Erro na Imagem": 1.0}

def predict_masked(image_dict):
    if not models_loaded or mlp_masked is None:
        return {"Modelo Mascarado não treinado": 1.0}
    try:
        img_scaled = process_image(image_dict)
        if img_scaled is None:
            return {"Vazio": 1.0}
        
        prob_mlp = mlp_masked.predict_proba(img_scaled)[0]
        classes = mlp_masked.classes_
        
        result = {}
        for idx, cls in enumerate(classes):
            result[str(cls)] = float(prob_mlp[idx])
            
        return result
    except Exception as e:
        print(f"Erro no predict_masked: {e}")
        return {"Erro na Imagem": 1.0}

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

* {
    font-family: 'Inter', sans-serif !important;
}

/* Esconder footers e logos do Gradio */
footer {display: none !important;}
.built-with {display: none !important;}

/* Fundo da aplicação (Estética Dark Premium) */
body, .gradio-container {
    background: linear-gradient(135deg, #121212 0%, #1a1a24 100%) !important;
    color: #e0e0e0;
}

/* Estilo do título */
h1 {
    text-align: center;
    background: -webkit-linear-gradient(45deg, #00f2fe, #4facfe);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800 !important;
    padding-bottom: 20px;
}

/* Suavizar abas e painéis com Glassmorphism leve */
.tabs {
    border-radius: 12px;
    background: rgba(30, 30, 40, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.05);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
}

.tabitem {
    padding: 25px !important;
    border: none !important;
}

/* Botões vibrantes */
button.primary {
    background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%) !important;
    border: none !important;
    box-shadow: 0 4px 15px rgba(0, 242, 254, 0.4) !important;
    transition: transform 0.2s ease;
}
button.primary:hover {
    transform: translateY(-2px);
}
"""

with gr.Blocks(title="MNIST Predictor Premium", css=custom_css, theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# 🧠 IA Preditiva - MNIST")
    
    if not models_loaded:
        gr.Markdown("⚠️ **AVISO:** Os modelos ainda não foram treinados. Rode `python src/train_and_save.py` primeiro!")
        
    with gr.Tabs():
        with gr.TabItem("1. Preditor Normal (Múltiplos Modelos)"):
            gr.Markdown("Escolha entre desenhar um número no quadro ou enviar uma foto sua, e veja a inferência em tempo real dos três modelos competindo!")
            
            with gr.Row():
                # Coluna Esquerda: Inputs
                with gr.Column(scale=1, min_width=350):
                    with gr.Tabs():
                        with gr.TabItem("🎨 Desenhar"):
                            sketchpad = gr.Sketchpad(label="Quadro Branco (Desenhe aqui)", height=350)
                            btn_sketch = gr.Button("🔮 Prever Desenho", variant="primary")
                        
                        with gr.TabItem("📁 Upload de Foto"):
                            image_input = gr.Image(type="numpy", image_mode="L", label="Envie a foto do seu papel", height=300)
                            btn_image = gr.Button("🔮 Prever Imagem", variant="primary")
                            
                            arquivos_exemplos = glob.glob("testes_manuscritos/*.png")
                            if arquivos_exemplos:
                                gr.Examples(examples=arquivos_exemplos, inputs=image_input, label="Exemplos Prontos")

                # Coluna Direita: Outputs
                with gr.Column(scale=2, min_width=500):
                    with gr.Row():
                        output_knn = gr.Label(num_top_classes=3, label="📊 Predição KNN")
                        output_xgb = gr.Label(num_top_classes=3, label="📊 Predição Gradient Boosting")
                        output_mlp = gr.Label(num_top_classes=3, label="📊 Predição MLP")

            btn_sketch.click(predict_digit, inputs=sketchpad, outputs=[output_knn, output_xgb, output_mlp])
            btn_image.click(predict_digit, inputs=image_input, outputs=[output_knn, output_xgb, output_mlp])

        with gr.TabItem("2. Desafios Acadêmicos (OOD)"):
            gr.Markdown("### Teste de Generalização Extrema (Overconfidence)")
            gr.Markdown("O modelo abaixo (MLP Masked) foi treinado de propósito **SEM as imagens dos números 0 e 9** (Class Masking). Ele não sabe que esses números existem.")
            gr.Markdown("Tente desenhar um `0` ou um `9` e veja como ele é forçado a mapear para as classes conhecidas, demonstrando extrema certeza matemática numa predição errada!")
            
            with gr.Row():
                with gr.Column(scale=1):
                    sketchpad_ood = gr.Sketchpad(label="Desenhe um 0 ou 9")
                    btn_ood = gr.Button("Forçar Predição (OOD)", variant="primary")
                with gr.Column(scale=1):
                    output_ood = gr.Label(num_top_classes=5, label="Alucinação da Rede (MLP Masked)")
            
            btn_ood.click(predict_masked, inputs=sketchpad_ood, outputs=output_ood)

if __name__ == "__main__":
    demo.launch(share=False)
