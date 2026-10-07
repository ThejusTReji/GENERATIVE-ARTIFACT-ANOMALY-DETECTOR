import os
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import ResNet50

# ── Ensure models directory exists ──────────────────────────────
os.makedirs("models", exist_ok=True)

# ── Data Paths ──────────────────────────────────────────────────
# The dataset should be extracted here:
DATASET_DIR = "dataset_generalized"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "test") # CIFAKE usually uses 'test' instead of 'validation'

if not os.path.exists(TRAIN_DIR):
    print(f"Error: Could not find {TRAIN_DIR}. Please make sure you extracted the CIFAKE dataset into 'dataset_generalized'.")
    exit(1)

# ── Data Generators ─────────────────────────────────────────────
# We process raw RGB pixels directly. No DCT conversion.
train_gen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=15,
    zoom_range=0.15,
    validation_split=0.2 # Use split if 'test' folder doesn't exist
)

val_gen = ImageDataGenerator(rescale=1./255)

print("Loading Training Data...")
train_data = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    color_mode="rgb"
)

# CIFAKE structure fallback logic
if os.path.exists(VAL_DIR):
    print("Loading Validation Data from test/ folder...")
    val_data = val_gen.flow_from_directory(
        VAL_DIR,
        target_size=(224, 224),
        batch_size=32,
        class_mode="binary",
        color_mode="rgb"
    )
else:
    print("No test folder found. Using 20% validation split from train...")
    # Re-initialize train_data with subset="training"
    train_data = train_gen.flow_from_directory(
        TRAIN_DIR,
        target_size=(224, 224),
        batch_size=32,
        class_mode="binary",
        color_mode="rgb",
        subset="training"
    )
    val_data = train_gen.flow_from_directory(
        TRAIN_DIR,
        target_size=(224, 224),
        batch_size=32,
        class_mode="binary",
        color_mode="rgb",
        subset="validation"
    )

# ── Model Architecture (Transfer Learning) ────────────────────────
# ResNet50 weights must be pre-downloaded using download_resnet_weights.py
print("\nBuilding ResNet50 Semantic Vision Model...")
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
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        "models/semantic_vision_best.h5",
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
print("\nStarting Training...")
history = model.fit(
    train_data,
    epochs=15, # CIFAKE converges quickly
    validation_data=val_data,
    callbacks=callbacks,
    verbose=1
)

# ── Save Final Model ─────────────────────────────────────────────
model.save("models/semantic_vision_model.h5")
print("\nModel saved to models/semantic_vision_model.h5")

# ── Plot Accuracy ─────────────────────────────────────────────────
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"],    label="Train")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("Accuracy (ResNet50 Semantic Vision)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"],    label="Train")
plt.plot(history.history["val_loss"], label="Validation")
plt.title("Loss (Semantic Vision)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("models/generalized_training_plot.png")
print("Saved training plot to models/generalized_training_plot.png")
