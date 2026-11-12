import torch
import torchvision
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt

# ==============================
# 1 - المسارات
# ==============================
data_dir = "data/data1a"
train_dir = os.path.join(data_dir, "training")
val_dir = os.path.join(data_dir, "validation")

# ==============================
# 2 - تجهيز الصور
# ==============================
transform = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

# ==============================
# 3 - تحميل الـ Dataset
# ==============================
train_data = datasets.ImageFolder(train_dir, transform=transform["train"])
val_data   = datasets.ImageFolder(val_dir,   transform=transform["val"])

train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
val_loader   = torch.utils.data.DataLoader(val_data,   batch_size=32, shuffle=False)

class_names = train_data.classes
print(f"الأصناف : {class_names}")

# ==============================
# 4 - النموذج EfficientNet
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"كيتستعمل : {device}")

model = models.efficientnet_b0(weights="IMAGENET1K_V1")

# تجميد الطبقات القديمة
for param in model.parameters():
    param.requires_grad = False

# تغيير آخر طبقة
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model = model.to(device)

# ==============================
# 5 - التدريب
# ==============================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

epochs = 10
train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(epochs):
    # Training
    model.train()
    running_loss, correct, total = 0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / len(train_loader)
    train_acc  = 100 * correct / total
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # Validation
    model.eval()
    running_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / len(val_loader)
    val_acc  = 100 * correct / total
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%")

# ==============================
# 6 - حفظ النموذج
# ==============================
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/car_damage_model.pth")
print("✅ النموذج تحفظ فـ models/car_damage_model.pth")

# ==============================
# 7 - رسم النتائج
# ==============================
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.legend()
plt.title("Loss")

plt.subplot(1, 2, 2)
plt.plot(train_accs, label="Train Acc")
plt.plot(val_accs, label="Val Acc")
plt.legend()
plt.title("Accuracy")

plt.savefig("models/training_results.png")
plt.show()
print("✅ النتائج تحفظوا فـ models/training_results.png")