import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Stop extra warnings

import tensorflow as tf
import random

base_dir = "./dataset"

BATCH_SIZE = 32 # number of training samples processed by the neural network before the model's weights are updated via backpropagation
IMG_HEIGHT = 256 # 2560 (has to be lowered)
IMG_WIDTH = 192  # 1920
SEED = random.randint(1, 9999) # Random seed to maintain identical results
print(f"SEED: {SEED}")

# Load 80% of the images for training
train_dataset = tf.keras.utils.image_dataset_from_directory(
    base_dir,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode="categorical",
)

# Load the remaining 20% of the images for validation
val_dataset = tf.keras.utils.image_dataset_from_directory(
    base_dir,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode="categorical",
)

# Confirm class names
class_names = train_dataset.class_names
print(f"Class names found: {class_names}")

# Define CNN
model = tf.keras.models.Sequential([

    # Rescaling layer shifts pixel values from [0, 255] to [0, 1]
    tf.keras.layers.Rescaling(1.0 / 255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', 
    metrics=['accuracy', 'precision', 'recall', 
    tf.keras.metrics.F1Score(average='weighted', name='f1_score')])

NUM_EPOCHS = 15

history = model.fit(
    train_dataset,
    validation_data = val_dataset,
    epochs = NUM_EPOCHS,
    verbose = 1,
    shuffle=False # Remove warning since dataset is already shuffled
)

# Classification Reports
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

results = model.evaluate(val_dataset, return_dict=True, verbose=0)

print("\nValidation Model Evaluation")
print(f"Accuracy:  {results['accuracy']:.4f}")
print(f"Precision: {results['precision']:.4f}")
print(f"Recall:    {results['recall']:.4f}")
print(f"F1-Score:  {results['f1_score']:.4f}")

# Extract aligned true labels and predictions from val_dataset
y_true = []
y_pred = [] # y as in y = f(x)

for images, labels in val_dataset:
    preds = model.predict(images, verbose=0) # Predictions happen after training
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

import matplotlib.pyplot as plt

# Plot train and test accuracies vs Epochs
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracies v.s Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()