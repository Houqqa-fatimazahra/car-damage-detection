from flask import Flask, render_template, request, jsonify
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import base64
import torch.nn as nn
import os
import json
from datetime import datetime
from price import estimate_price

app = Flask(__name__)

# ==============================
# تحميل النموذج
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(torch.load("models/car_damage_model.pth", map_location=device))
model.eval()
model = model.to(device)
class_names = ['Endommagée', 'Intacte']

HISTORY_FILE = "history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_history(entry):
    history = load_history()
    history.append(entry)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f)

# ==============================
# تحليل الصورة
# ==============================
def analyze_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)

    gradients = []
    activations = []

    def save_gradient(grad):
        gradients.append(grad)

    def forward_hook(module, input, output):
        activations.append(output)
        output.register_hook(save_gradient)

    target_layer = model.features[-1]
    handle = target_layer.register_forward_hook(forward_hook)

    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()
    confidence = torch.softmax(output, dim=1)[0][pred_class].item()

    model.zero_grad()
    output[0][pred_class].backward()
    handle.remove()

    grad = gradients[0].cpu().detach().numpy()[0]
    act = activations[0].cpu().detach().numpy()[0]
    weights = grad.mean(axis=(1, 2))
    cam = np.zeros(act.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * act[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    img_resized = np.array(img.resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.6 * img_resized + 0.4 * heatmap).astype(np.uint8)

    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    gradcam_b64 = base64.b64encode(buffer).decode('utf-8')

    return class_names[pred_class], confidence, gradcam_b64

# ==============================
# Routes
# ==============================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['image']
    path = f"static/temp_{file.filename}"
    file.save(path)

    label, confidence, gradcam_b64 = analyze_image(path)
    price = estimate_price(label, confidence)

    entry = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "resultat": label,
        "confiance": f"{confidence*100:.1f}%",
        "prix": f"{price['prix_min']} — {price['prix_max']} DH",
        "statut": price['statut']
    }
    save_history(entry)

    return jsonify({
        "label": label,
        "confidence": f"{confidence*100:.1f}%",
        "gradcam": gradcam_b64,
        "price": price
    })

@app.route('/history')
def history():
    return jsonify(load_history())

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    msg = data.get('message', '').lower()
    label = data.get('label', '')
    confidence = data.get('confidence', '')
    price = data.get('price', {})

    if any(w in msg for w in ['grave', 'sérieux']):
        reply = f"Le dommage détecté est : {price.get('statut','')}, avec une confiance de {confidence}."
    elif any(w in msg for w in ['prix', 'coût', 'tarif']) or (('combien' in msg) and ('temps' not in msg) and ('durée' not in msg)):
        reply = f"Le coût estimé de réparation est : {price.get('prix_min')} — {price.get('prix_max')} DH."
    elif any(w in msg for w in ['conseil', 'recommandation', 'que faire']):
        reply = f"Notre recommandation : {price.get('recommandation','')}"
    elif any(w in msg for w in ['résultat', 'diagnostic', 'analyse']):
        reply = f"Résultat : {label} avec {confidence} de confiance."
    elif any(w in msg for w in ['assurance', 'couverture', 'remboursement']):
        if label == "Endommagée":
            reply = "Avec une assurance tous risques, les dommages sont généralement couverts. Contactez votre assureur avec les photos et ce rapport d'analyse."
        else:
            reply = "Votre véhicule est en bon état, aucune déclaration d'assurance nécessaire."
    elif any(w in msg for w in ['garage', 'réparateur', 'atelier', 'mécanicien']):
        if label == "Endommagée":
            reply = f"Nous vous recommandons de consulter un carrossier agréé. Coût estimé : {price.get('prix_min')} — {price.get('prix_max')} DH."
        else:
            reply = "Votre véhicule est intact, aucune visite en garage nécessaire."
    elif any(w in msg for w in ['délai', 'temps', 'durée', 'combien de temps']):
        if label == "Endommagée":
            if price.get('prix_min', 0) >= 10000:
                reply = "Pour des dommages sévères, la réparation peut prendre 2 à 4 semaines."
            elif price.get('prix_min', 0) >= 3000:
                reply = "Pour des dommages modérés, comptez 3 à 7 jours de réparation."
            else:
                reply = "Pour des dommages légers, la réparation prend généralement 1 à 2 jours."
        else:
            reply = "Aucune réparation nécessaire, votre véhicule est en parfait état."
    elif any(w in msg for w in ['dangereux', 'rouler', 'conduire', 'sécurité']):
        if label == "Endommagée" and price.get('prix_min', 0) >= 10000:
            reply = "⚠️ Attention ! Les dommages sont sévères. Il est déconseillé de conduire ce véhicule avant réparation."
        elif label == "Endommagée":
            reply = "Les dommages sont présents mais vous pouvez rouler prudemment. Une réparation rapide est recommandée."
        else:
            reply = "✅ Votre véhicule est en bon état, vous pouvez rouler en toute sécurité."
    else:
        reply = "Je suis votre assistant d'analyse de dommages. Vous pouvez me poser des questions sur : le résultat, le prix, l'assurance, le garage, le délai ou la sécurité."

    return jsonify({"reply": reply})

if __name__ == '__main__':
    app.run(debug=True)