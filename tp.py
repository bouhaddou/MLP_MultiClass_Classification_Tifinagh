import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam


# 1. Chargement des données

data_dir = os.path.join(os.getcwd(), 'amhcd-data-64/tifinagh-images/')
print("Chemin vers les données :", data_dir)

try:
    csv_path = os.path.join(data_dir, 'labels-map.csv')
    labels_df = pd.read_csv(csv_path)
    assert 'image_path' in labels_df.columns and 'label' in labels_df.columns, "Le CSV doit contenir 'image_path' et 'label'"
except FileNotFoundError:
    print("labels-map.csv introuvable. Construction à partir des dossiers...")
    image_paths, labels = [], []
    for label_dir in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label_dir)
        if os.path.isdir(label_path):
            for img_name in os.listdir(label_path):
                image_paths.append(os.path.join(label_path, img_name))
                labels.append(label_dir)
    labels_df = pd.DataFrame({"image_path": image_paths, "label": labels})

print("Nombre total d’images :", len(labels_df))
print(labels_df.head())

# 2. Prétraitement des images  

def load_and_preprocess_image(path, size=(32, 32)):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, size)
    img = img.astype("float32") / 255.0
    return img.flatten()  # aplatie en vecteur

X = np.array([load_and_preprocess_image(os.path.join(data_dir, p)) for p in labels_df['image_path']])

# Encodage labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels_df['label'])

# Split Train/Val/Test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)

# One-Hot encoding
one_hot_encoder = OneHotEncoder(sparse_output=False)
y_train_one_hot = one_hot_encoder.fit_transform(y_train.reshape(-1, 1))
y_val_one_hot   = one_hot_encoder.transform(y_val.reshape(-1, 1))
y_test_one_hot  = one_hot_encoder.transform(y_test.reshape(-1, 1))

num_classes = y_train_one_hot.shape[1]

print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
print("Nombre de classes :", num_classes)

# 3. Modèle TensorFlow


model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(num_classes, activation="softmax")
])

model.compile(optimizer=Adam(learning_rate=0.01),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# 4. Entraînement

history = model.fit(
    X_train, y_train_one_hot,
    validation_data=(X_val, y_val_one_hot),
    epochs=100,
    batch_size=32,
    verbose=1
)


# 5. Évaluation

test_loss, test_acc = model.evaluate(X_test, y_test_one_hot, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

# Prédictions
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\nRapport de classification :")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('Matrice de confusion (Test set)')
plt.xlabel('Prédit')
plt.ylabel('Réel')
plt.show()

# 6. Courbes

plt.figure(figsize=(12, 5))

# Loss
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Courbe de perte")
plt.xlabel("Époque")
plt.ylabel("Perte")
plt.legend()

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.title("Courbe de précision")
plt.xlabel("Époque")
plt.ylabel("Précision")
plt.legend()

plt.tight_layout()
plt.show()
