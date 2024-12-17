import pandas as pd
import numpy as np
from PIL import Image
import os

csv_path = 'data/data.csv'
image_dir = 'data/images/'

# Load CSV
data = pd.read_csv(csv_path)
# Clean column names by stripping leading/trailing spaces or quotes
data.columns = data.columns.str.strip().str.replace('"', '').str.replace("'", '')

# Check the column names and data
print("CSV Columns:", data.columns)
print("First rows of data:\n", data.head())
print("Unique values in 'note' column:", data['note'].unique())
# Function to load and preprocess images
def load_images(image_dir, data, img_size=(64, 64)):
    images = []
    labels = []
    for _, row in data.iterrows():
        img_path = os.path.join(image_dir, f"{row['name']}")
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        img = img.resize(img_size)  # Resize to a fixed size
        images.append(np.array(img) / 255.0)  # Normalize pixel values
        labels.append(row[1])
    print("Number of images loaded:", len(images))
    print("Number of labels:", len(labels))
    return np.array(images), np.array(labels)

# Load and preprocess
X, y = load_images(image_dir, data)
print(X.shape, y.shape)
print("X head--\n", X[:5])
print("y head--\n", y[:5])
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Encode labels to integers using LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Convert labels to integers

# Convert to one-hot encoding using pandas
y_one_hot = pd.get_dummies(y_encoded).values  # Convert to numpy array

print("Classes:", label_encoder.classes_)
print("One-Hot Encoded Labels Shape:", y_one_hot.shape)

from sklearn.model_selection import train_test_split

# Split data into training and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

from tensorflow.keras import models, layers

# Build the CNN model
model = models.Sequential()

# Add convolutional layers
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
model.add(layers.MaxPooling2D((2, 2)))

# Add another convolutional layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Flatten the output of the convolutional layers
model.add(layers.Flatten())

# Add a fully connected layer
model.add(layers.Dense(128, activation='relu'))

# Output layer with softmax activation for classification
model.add(layers.Dense(y_one_hot.shape[1], activation='softmax'))  # Number of classes is the number of distinct notes

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()  # Print the model architecture

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Print training and validation accuracy for each epoch
print(f"Training accuracy: {history.history['accuracy'][-1]}")
print(f"Validation accuracy: {history.history['val_accuracy'][-1]}")

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy}")

# Predict on a new image
import random
def test_random_images(X, y, model, label_encoder, data, num_tests=5, img_size=(64, 64)):
    for _ in range(num_tests):
        # Pick a random index
        random_index = random.randint(0, len(X) - 1)

        # Get the image, true label, and file name from the data
        img = X[random_index]
        true_label = y[random_index]
        file_name = data.iloc[random_index]['name']  # Assuming 'name' column contains file names

        # Convert the image array back to an image object for display or further processing (if necessary)
        img = Image.fromarray((img * 255).astype(np.uint8)).resize(img_size)
        
        # Normalize the image again if needed (depending on how your model was trained)
        img = np.expand_dims(img, axis=0)  # Add batch dimension for the model input
        img = img / 255.0  # Normalize pixel values if required

        # Make prediction
        predicted_class = model.predict(img)
        predicted_label = label_encoder.inverse_transform(np.argmax(predicted_class, axis=1))

        # Compare predicted note to the true note
        print(f"File: {file_name}, True note: {true_label}, Predicted note: {predicted_label[0]}")

# Example of testing 5 random images
test_random_images(X, y, model, label_encoder, data, num_tests=5)

