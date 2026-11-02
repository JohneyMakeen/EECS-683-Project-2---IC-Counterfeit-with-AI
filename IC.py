import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn.metrics

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

TRAIN_DIRECTORY_LOCATION = 'Train'
TEST_DIRECTORY_LOCATION = 'Test'
TARGET_SIZE = (500, 500)
CLASS_MODE = 'categorical'

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIRECTORY_LOCATION,
    target_size = TARGET_SIZE,
    batch_size = 4,
    class_mode = CLASS_MODE
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIRECTORY_LOCATION,
    target_size = TARGET_SIZE,
    batch_size = 4,
    class_mode = CLASS_MODE,
    shuffle = False
)

print("Class mapping:", train_generator.class_indices)
print("Test class mapping:", test_generator.class_indices)

# Define CNN
model = tf.keras.models.Sequential([
  tf.keras.layers.Input(shape=(500,500, 3)),
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

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

NUM_EPOCHS = 5

history = model.fit(
      train_generator,
      epochs = NUM_EPOCHS,
      verbose = 1)

# Generate predictions for the test dataset
test_generator.reset()
predictions = model.predict(test_generator)

# Obtain predicted class labels and true class labels
y_pred = predictions.argmax(axis=1)
y_true = test_generator.classes

# Calculate evaluation metrics
accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
precision = sklearn.metrics.precision_score(y_true, y_pred, pos_label=1, zero_division=0)
recall = sklearn.metrics.recall_score(y_true, y_pred, pos_label=1, zero_division=0)
f1 = sklearn.metrics.f1_score(y_true, y_pred, pos_label=1, zero_division=0)

# Display
print('\nBaseline Detection performance')
print('--------------------------------')
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-Score: {f1:.4f}')

# Confusion Matrix
cm = sklearn.metrics.confusion_matrix(y_true, y_pred, labels=[0,1])
print('\nConfusion Matrix:')
print(cm)

# Classification Report
print('\nClassification Report:')
cr = sklearn.metrics.classification_report(y_true, y_pred, target_names=['Authentic', 'Counterfeit'], zero_division=0)
print(cr)

# Plot train and test accuracies vs Epochs
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.title('Training Accuracy vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()