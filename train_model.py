import os
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    BatchNormalization
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import ResNet50

# ── Ensure models directory exists ──────────────────────────────
os.makedirs("models", exist_ok=True)

# ── Data Generators ─────────────────────────────────────────────
train_gen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,        # augmentation
    rotation_range=10,
    zoom_range=0.1
)

val_gen = ImageDataGenerator(
    rescale=1./255
)

train_data = train_gen.flow_from_directory(
    "dct_data/train",
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    color_mode="rgb"
)

val_data = val_gen.flow_from_directory(
    "dct_data/validation",
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    color_mode="rgb"
)

# ── Model Architecture (Transfer Learning) ────────────────────────
base_model = ResNet50(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # Freeze base model

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ── Callbacks ────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        "models/fake_detector_best.h5",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        verbose=1,
        min_lr=1e-6
    )
]

# ── Training ─────────────────────────────────────────────────────
history = model.fit(
    train_data,
    epochs=10,
    validation_data=val_data,
    callbacks=callbacks,
    verbose=1
)

# ── Save Final Model ─────────────────────────────────────────────
model.save("models/fake_detector.h5")
print("\nModel saved to models/fake_detector.h5")

# ── Plot Accuracy ─────────────────────────────────────────────────
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"],    label="Train")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("Accuracy — ResNet50 DCT Model")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"],    label="Train")
plt.plot(history.history["val_loss"], label="Validation")
plt.title("Loss — ResNet50 DCT Model")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("models/training_plot.png")
plt.show()

# ── Test Evaluation ───────────────────────────────────────────────
test_gen = ImageDataGenerator(rescale=1./255)

test_data = test_gen.flow_from_directory(
    "dct_data/test",
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    color_mode="rgb"
)

loss, accuracy = model.evaluate(test_data, verbose=1)
print(f"\nTest Loss    : {loss:.4f}")
print(f"Test Accuracy: {accuracy*100:.2f}%")
