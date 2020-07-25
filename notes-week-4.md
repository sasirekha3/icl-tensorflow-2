
# Saving and Loading Models

Can save and load models. Or save only architecture. Continuing an interrupted training later. 

The latter is possible through callbacks.

Using pre-trained TF2 models.

### Saving and Loading Model Weights (aka parameters)

Using a built-in callback called `Model checkpoint Callback`.

For now, saving only weights.

```
model = Sequential([
    Dense(64, activation='sigmoid', input_shape=(10, )),
    Dense(1)
])

model.compile (optimizer='sgd', loss=BinaryCrossentropy(from_logits=True))

checkpoint = tf.keras.callbacks.ModelCheckpoint('my_model', save_weights_only=True)

model.fit(X_train, y_train, epochs=10, callbacks=[checkpoint])
```

The first argument for ModelCheckpoint() is the filepath to where the model weights are to be saved. These weights are by default stored in the native Tensorflow format. The filepath argument is also the only required argument.
The following files are created when saving with native TF format.

```
checkpoint
my_model.data-00000of00001
my_model.index
```
These files are overwritten upon re-training, but that can be changed.


If the filepath is given a `.h5` extension, then they are saved using the Keras HDF5 format.


##### Loading the model

```
model.load_weights('my_model')

# or

model.load_weights('keras_weights.h5')
```

##### Manually saving model weights

```
model = Sequential([
    Dense(64, activation='sigmoid', input_shape=(10, )),
    Dense(1)
])

model.compile (optimizer='sgd', loss='mse', metrics=['mae'])

early_stopping = EarlyStopping(monitor='val_mae', patience=2)

model.fit(X_train, y_train, validation_split=0.2, epochs=50, callbacks=[early_stopping])

model.save_weights('my_model')
```


### Saving Model

Other features of the ModelCheckpoint callback

```
model = Sequential([
    Dense(16, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')
])

model.compile (optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['acc', 'mae'])

checkpoint = tf.keras.callbacks.ModelCheckpoint('training_run_1/my_model', save_weights_only=True, save_best_only=True, monitor='val_loss', mode='max')

model.fit(X_train, y_train, validation_data=(X_val, y_val)), epochs=10, callbacks=[checkpoint], batch_size=16)

```

Params of ModelCheckpoint:
- `save_freq='epoch'`: default is the string 'epoch'. Can also set it to an integer value which is the number of samples seen by the model since the last time the weights were saved => number of samples seen as opposed to number of iterations.
- `save_best_only=False`: False is default. Setting to `True` will only save weights that satisfy a certain criteria. This criteria is set by using the `mode` and `monitor` arguments.
- `monitor='val_loss'`: default is 'val_loss'. This is the criterion that will be monitored to decide whether to save that set of weights or not.
- `mode='max'`: default is 'auto'. Tells whether we want to maximize or minimize the measure we are monitoring.
- *filename*: The first argument. We can make the filename include information about the epoch and batch using - `training_run_1/my_model.{epoch}.{batch}` or `training_run_1/my_model.{epoch}-{val_loss:.4f}`


### Saving Model Architecture as well

```
model = Sequential([
    Dense(16, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')
])

model.compile (optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['acc', 'mae'])

checkpoint = tf.keras.callbacks.ModelCheckpoint('training_run_1/my_model', save_weights_only=False, save_best_only=True, monitor='val_loss', mode='max')

model.fit(X_train, y_train, validation_data=(X_val, y_val)), epochs=10, callbacks=[checkpoint], batch_size=16)
```

The argument `save_weights_only` is set to `false`, which is default. 

This leads to the creation of the following folder structure:

```
my_model/assets
my_model/saved_model.pb
my_model/variables/variables.data-00000-of-00001
my_model/variables/variables.index
```

Variables folder contains saved weights of the model. The `.pb` file saves the model arch, i.e., model graph.

Assets folder will be used to store additional files needed for the graph (empty in this simple example).

As always, we can use the keras `.h5` extension which only creates one h5 file that contains the weights and the architecture.

Alternatively, we can use:

```
model.save('my_model')
```

or 

```
model.save('keras_model.h5')
```

##### Loading model architecture with weights

Use:

```
new_model = load_model('my_model') # or .h5 for keras HDF5
```

We can now run `.summary()`, `.fit()`, `.evaluate()`, and `.predict()`.


### Pre-trained Keras models

keras.io -> ImageNet dataset + examples

```
from tensorflow.keras.application.resnet50 import ResNet50

model = ResNet50(weights='imagenet', include_top=False) #~/.keras/models/
```

When the above line is executed for the first time, models have to be downloaded to your local system. 

`include_top` is set to `True` by default, which means that the entire model is loaded and in. But setting it to `False` will load the model without the fully-connected layer at the top of the model so that it can be used for transfer learning applications.

Making predictions from a downloaded model:

```
from tensorflow.keras.application.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

model = ResNet50(weights='imagenet', include_top=False)

img_input = image.load_img('my_picture.jpg', target_size=(224, 224))
img_input = image.img_to_array(img_input)
img_input = preprocess_input(img_input[np.newaxis, ...])

preds = model.predict(img_input)
decoded_predictions = decode_predictions(preds, top=3)[0] #returns class, an english description, and probability
```

Keras' preprocessing module has an image class that allows for adjusting image sizes using `target_size`. `(224, 224)` is the shape expected by resnet50. The model takes a numpy array as input which is accomplished by `img_to_array()`. To preprocess the image using the same parameters used during training is done using `preprocess_input()`. A dummy dimension is added as well.

To map predictions to classes, we use `decode_predictions()`. This returns class, an english description, and probability.


### Tensorflow Hub

A little different from pretrained models from Keras.

Offers reusable ML modules as separate library. 

There are some complete models or modules we can use as part of bigger model.

