### Keras Sequential API

High level NN API. Developed without a backend and may adopt different backends. TF2 adopted Keras as default.

namespace - `tf.keras`
use tf docs as goto reference

### Building a Feedforward networks (multi-layer perceptrons)
```
model = Sequential([
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
```

*with sizes at the time of creation*
```
model = Sequential([
    Dense(64, activation='relu', input_shape=(784,)),
    Dense(10, activation='softmax')
])
```

or

```
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(784,)))
model.add(Dense(10, activation='softmax'))
```

*Flatten and Dense*
```
model = Sequential([
    Flatten(input_shape=(28,28)), # (784, ) => 1D vector
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
```

### Building a CNN
`Conv2D(no_of_filters, (kernel_dims), ...)`
`MaxPooling2D((pooling layer dims))`

```
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(32, 32, 3)), # (None, 30, 30, 16) because 16 filters and 'VALID' padding (aka no-zero padding)
    MaxPooling2D((3,3)), # Non overlapping => (None, 10, 10, 16)
    Flatten(), # (None, 1600) => 1D vector from 10 * 10 * 16
    Dense(64, activation='relu'), # (None, 64)
    Dense(10, activation='softmax') # (None, 10)
])
```

*with SAME padding*

```
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(32, 32, 3)), # (None, 32, 32, 16) because 16 filters and 'SAME' padding adds padding to keep the h and w of the input
    MaxPooling2D((3,3)), # Non overlapping => (None, 10, 10, 16)
    Flatten(), # (None, 1600) => 1D vector from 10 * 10 * 16
    Dense(64, activation='relu'), # (None, 64)
    Dense(10, activation='softmax') # (None, 10)
])
```

Pooling layers are used to:
1) Allow a degree of translational invariance on the input
2) Downsample the spatial dimensions, thereby reducing the number of network parameters

*notebook notes*
- Setting strides=2 sets the stride to 2 in every dimension in Conv2D
- Adding padding='SAME' keeps the dims same as the input shape in Conv2D
- By default, the data_format is 'channles_last', but it can also be 'channels_first' 


### The Compile Method

```
model = Sequential()
model.add(Dense(64, activation='elu', input_shape=(32,)))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    optimizer='sgd', # can also use adam, rmsprop, adadelta
    loss='binary_crossentropy', # can also use mean_squared_error, categorical_crossentropy
    metrics=['accuracy', 'mae'] 
)
```

The compile function can also be used as:

```
model = Sequential()
model.add(Dense(64, activation='elu', input_shape=(32,)))
model.add(Dense(1, activation='linear')) # linear is default activation

model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True), # can also use adam, rmsprop, adadelta
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), # can also use mean_squared_error, categorical_crossentropy
    metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.7), tf.keras.metrics.MeanAbsoluteError()] 
)
```

### The fit method

```
model = Sequential()
model.add(Dense(64, activation='elu', input_shape=(32,)))
model.add(Dense(100, activation='softmax'))

model.compile(
    optimizer='rmsprop', # can also use adam, rmsprop, adadelta
    loss='sparse_categorical_crossentropy', # can also use mean_squared_error, categorical_crossentropy
    metrics=['accuracy'] 
)

history = model.fit(X_train, y_train, epochs=10, batch_size=16) # default epochs=1; divide current single array into batches of 16
# X_train dims: (num_samples, num_features)
# y_train dims: (num_samples, num_classes)
# history: a callback that contains a history of the training of the model

# Sparse representation, i.e., single integer for each class => sparse_categorical_crossentropy
# Labels are represented as a 1-hot array => use categorical_crossentropy loss
```


### The evaluate and predict methods

Evaluate how well the network has learned and provide unseen input data (i.e., not train_data)

```
model = Sequential([
    Dense(1, activation='sigmoid', input_shape=(12,))
])

model.compile(
    optimizer='sgd',
    loss='binary_crossentropy',
    metrics=['accuracy', 'mae']
)

model.fit(X_train, y_train)

loss, accuracy, mae = model.evaluate(X_test, y_test) # if we had given more metrics, then they will all be returned

# X_sample dims: (num_samples, 12) # num_features corresponds to the input shape

pred = model.predict(X_sample) # returns an array of an array of size num_samples => 2D array

```

#### eg 1:

```
model = Sequential([
    Dense(1, activation='sigmoid', input_shape=(12,))
])

model.compile(
    optimizer='sgd',
    loss='binary_crossentropy',
    metrics=['accuracy', 'mae']
)

model.fit(X_train, y_train)

loss, accuracy, mae = model.evaluate(X_test, y_test)

# X_sample dims: (2, 12), y_sample dims: (1, 12)

pred = model.predict(X_sample, y_sample) # [[output_probability_1], [output_probability_2]]

```

#### eg 2: 

```
model = Sequential([
    Dense(3, activation='sigmoid', input_shape=(12,)) # FInal layer size has to match y_sample's dim[0], which is num_classes
])

model.compile(
    optimizer='sgd',
    loss='binary_crossentropy',
    metrics=['accuracy', 'mae']
)

model.fit(X_train, y_train)

loss, accuracy, mae = model.evaluate(X_test, y_test)

# X_sample dims: (2, 12), y_sample dims:(2, 3)

pred = model.predict(X_sample, y_sample) # [[output_probability_1_1/3, output_probability_2/3, output_probability_3/3], 
                                  [output_probability_2_1/3, output_probability_2_2/3, output_probability_2_3/3]]

# output_probability_1_1/3 + output_probability_2/3 + output_probability_3/3 = 1; and likewise for all other arrays of output_probabilities

```

output dims: (num_samples, num_classes) => same as y_train


## Programming Assignment

Classification model for the MNIST handwritten digits dataset

