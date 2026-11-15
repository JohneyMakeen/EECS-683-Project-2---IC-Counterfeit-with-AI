import tensorflow as tf
import random

base_dir = "./dataset"

BATCH_SIZE = 32 # number of training samples processed by the neural network before the model's weights are updated via backpropagation
IMG_HEIGHT = 256 # 2560 (has to be lowered)
IMG_WIDTH = 192  # 1920
SEED = random.randint(1, 9999) # Random seed to maintain identical results

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

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'precision', 'recall'])

NUM_EPOCHS = 10

history = model.fit(
    train_dataset,
    validation_data = val_dataset,
    epochs = NUM_EPOCHS,
    verbose = 1)

import matplotlib.pyplot as plt

# Plot train and test accuracies vs Epochs
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracies v.s Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()