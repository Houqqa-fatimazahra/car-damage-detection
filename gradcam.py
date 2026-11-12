import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch.nn as nn

# ==============================
# 1 - Chargement du modèle
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(torch.load("models/car_damage_model.pth", map_location=device))
model.eval()
model = model.to(device)

class_names = ['Endommagée', 'Intacte']

# ==============================
# 2 - Prétraitement de l'image
# ==============================
def preprocess(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0).to(device), img

# ==============================
# 3 - Grad-CAM
# ==============================
def grad_cam(image_path):
    input_tensor, original_img = preprocess(image_path)

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

    img_resized = np.array(original_img.resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.6 * img_resized + 0.4 * heatmap).astype(np.uint8)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(img_resized)
    plt.title("Image Originale")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap)
    plt.title("Grad-CAM Heatmap")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title(f"Résultat : {class_names[pred_class]}\nConfiance : {confidence*100:.1f}%")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("models/gradcam_result.png")
    plt.show()
    print(f"✅ Résultat : {class_names[pred_class]} — Confiance : {confidence*100:.1f}%")

# ==============================
# 4 - Test sur une image
# ==============================
grad_cam("data/data1a/validation/00-damage/0001.JPEG")