import tensorflow as tf
from tensorflow.keras import layers, models
from datasets import load_dataset
import os

fashiondata=tf.keras.datasets.mnist
# Get the current directory
current_directory = os.getcwd()

# Create a new folder named "model_output" in the current directory
model_folder = os.path.join(current_directory, 'model_output')

os.makedirs(model_folder, exist_ok=True)

# Now, you can use 'model_folder' as the destination for saving your model files.

# Load the dataset
#dataset = load_dataset("miladfa7/Brain-MRI-Images-for-Brain-Tumor-Detection")
dataset = load_dataset("benschill/brain-tumor-collection", trust_remote_code=True)
# Function to preprocess data (adjust based on your dataset structure)
def preprocess_data(data):
    features = []
    labels = []
    for feat in data:
        image = feat['image']
        label = feat.get('label')
        if label is not None:
            features.append(image)
            labels.append(label)

    return features, labels

train_data = preprocess_data(dataset['train'])

def preprocess_image(img):
    # Add your image preprocessing steps (e.g., resize, normalization)
    # Example: resizing images to (224, 224)
    img = tf.image.resize(img, (224, 224))
    img = img / 255.0  # Normalize pixel values to [0, 1]
    return img

# Preprocess the data
train_data = preprocess_data(dataset['train'])
test_data = preprocess_data(dataset['test'])

# Continue with the rest of your code (model definition, compilation, training, and evaluation)
# ...

# Define the CNN model architecture
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_data['images'], train_data['labels'], epochs=10, validation_data=(test_data['images'], test_data['labels']))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_data['images'], test_data['labels'])
print(f'Test Accuracy: {test_accuracy}')
