import numpy as np
import json
import tensorflow as tf
from keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as mat
import keras

# Load and preprocess the data
output_data = []
with open("output.json", "r") as file:
    input_data = json.load(file)



for z in range(19445):
    output_data.append(1)

with open("nondrowsy.json", "r") as file:
    input_data += json.load(file)

for z in range(19445):
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



# Convert input data to TensorFlow tensors
X_train_tf = tf.convert_to_tensor(X_train, dtype=tf.float32)
Y_train_tf = tf.convert_to_tensor(Y_train, dtype=tf.float32)
X_val_tf = tf.convert_to_tensor(X_val, dtype=tf.float32)
Y_val_tf = tf.convert_to_tensor(Y_val, dtype=tf.float32)

model = keras.models.Sequential([
    # Input Layer with L2 Regularization
    keras.layers.Dense(16, activation='relu', input_shape=(3,),
                       kernel_regularizer=keras.regularizers.l2(0.01)),
    
    # Hidden Layers with L2 Regularization, Batch Normalization, and Dropout
    keras.layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.4),
    
    keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.4),
    
    keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.4),
    
    keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.4),
    
    keras.layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.4),
    
    # Output Layer
    keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=keras.regularizers.l2(0.01))  # Output layer for binary classification
])
early_stopping = tf.keras.callbacks.EarlyStopping(patience= 20)
lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor= 0.999,
    patience= 5,
    min_lr=0.000000001)



initial_learning_rate = 0.00003  # Set your desired initial learning rate here

# Define the optimizer with the initial learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)




# Compile the model
model.compile(optimizer= optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train_tf,
    Y_train_tf,
    epochs = 1000,
    batch_size=500,
    validation_data=(X_val, Y_val),
    callbacks= [lr_callback, early_stopping])

# Evaluate the model on the validation set
val_loss, val_accuracy = model.evaluate(X_val_tf, Y_val_tf)
print("Validation Loss:", val_loss)
print("Validation Accuracy:", val_accuracy)

# Save the trained model
model.save('1000epochs.h5')




# Print test accuracy

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
val_loss = history.history['val_loss']
loss = history.history['loss']
epochs = range(1, len(accuracy) + 1)




# Plotting accuracy across epochs
# mat.plot(epochs, accuracy, 'bo', label='Training accuracy')
# mat.plot(epochs, val_accuracy, 'r', label='Validation accuracy')
mat.plot(epochs, loss, 'r', label='training loss')
mat.plot(epochs, val_loss, 'g', label='Validation loss')
mat.title('Training and validation loss')
mat.xlabel('Epochs')
mat.ylabel('Loss')
mat.legend()
mat.show()

mat.plot(epochs, accuracy, 'r', label='training accuracy')
mat.plot(epochs, val_accuracy, 'g', label='validation accuracy')
mat.title('Training and validation accuracy')
mat.xlabel('Epochs')
mat.ylabel('Loss')
mat.legend()
mat.show()

print('some random shit')