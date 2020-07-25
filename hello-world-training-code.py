# Import TensorFlow

import tensorflow as tf

# Check its version

tf.__version__

# Train a feedforward neural network for image classification

import numpy as np

print('Loading data...\n')
data = np.loadtxt('./data/mnist.csv', delimiter=',')
print('MNIST dataset loaded.\n')

x_train = data[:, 1:]
y_train = data[:, 0]
x_train = x_train/255.

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(16, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print('Training model...\n')
model.fit(x_train, y_train, epochs=3, batch_size=32)

print('Model trained successfully!')