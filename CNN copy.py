import numpy as np
import json
import tensorflow as tf
from keras import layers, models
from sklearn.model_selection import train_test_split

# Load and preprocess the data
output_data = []
with open("output.json", "r") as file:
    input_data = json.load(file)

for _ in range(19445):
    output_data.append(1)

with open("nondrowsy.json", "r") as file:
    input_data += json.load(file)

for _ in range(19445):
    output_data.append(0)

filtered_input_data = []
filtered_output_data = []

for elem in input_data:
    if isinstance(elem, (list, tuple)) and len(elem) == 3:
        filtered_input_data.append(elem)
        filtered_output_data.append(output_data[len(filtered_input_data) - 1])

# Convert filtered input data to NumPy arrays
input_data_array = np.array(filtered_input_data)
output_data_array = np.array(filtered_output_data)

# Split the data into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(input_data_array, output_data_array, test_size=0.2, random_state=42)

# Define the CNN model
model = models.Sequential([
    layers.Conv1D(32, 3, activation='relu', input_shape=(3, 1)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_data=(X_val, Y_val))

# Evaluate the model on the validation set
val_loss, val_accuracy = model.evaluate(X_val, Y_val)
print("Validation Loss:", val_loss)
print("Validation Accuracy:", val_accuracy)

# Save the trained model
model.save('nopooling.h5')
