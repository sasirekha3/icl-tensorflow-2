# Week 3 - Validation, regularization, and callbacks

Additional options - want to generalize beyond exact data given to that model. This is called regularization.

Callbacks can be used to perform certain operations during training. 

Iris dataset.

### Validation sets

```
model = Sequential()
model.add(Dense(128, activation='tanh))
model.add(Dense(2))

opt = Adam(learning_rate = 0.05)
model.compile(optimizer=opt, loss='mse', metrics=['mape'])

history = model.fit(inputs, targets, validation_split=0.2)

```

Validation set is held out.

In model.fit, we can use `validation_split` to specify the % of data that needs to be held out.

The history object, when validation_split is used, will also record performance the validation set.

If we want to give the model already split data, then we can use `validation_data` keyword arg.

```
history = model.fit(inputs, targets, validation_data=(X_test, y_test))
```

When either `validation_*` keyword arg is used, the history object contains all the attributes it does for the training set for the validation set with the `val_*` naming convention.

For example, for the above model, `history.history` contains the following keys: `['loss', 'mape', 'val_loss', 'val_mape']`

We can use sklearn's libraries to split data into train, validation, and test sets:
```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=p1)
model.fit(X_train, y_train, validation_split=p2)
```

Here, p1 is a fraction of all data, but p2 is fraction of training data. So to have a 60/20/20 ration of training/validation/test sets, `p1=0.2` and `p2=0.25`.


### Regularization Techniques

##### L2 - weight decay

Weight matrices are sometimes called kernels. Therefore, a kernel_regularizer is for regularizing weights.

L2 Regularization, or weight decay, has a coefficient that multiplies the sum of squared weights in this layer. `0.001` is the weight decay coefficient.

```
model = Sequential([
    Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['acc'])
model.fit(inputs, targets, validation_split=0.25)
```

When the model is compiled, the weight decay penalty term is automatically added to the loss function. Here, the loss function becomes `binary_crossentropy + ((sum of squared weights of the first layer of the model) * weight_decay_coefficient)`

The penalty term is `(sum of squared weights of the first layer of the model)`

It has the effect of penalizing large values of weights, which keeps the model from overfitting the training data.


##### L1 regularization

```
model = Sequential([
    Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.005)),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['acc'])
model.fit(inputs, targets, validation_split=0.25)
```

The penalty term is `(sum of absolute weights of the first layer of the model)`

The regularization coefficient here is `0.005`. 

L1 generally makes the weights matrix more sparse, i.e., sets some of those values to zero. 

##### Using L1 and L2

```
model = Sequential([
    Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.001, l2=0.005)),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['acc'])
model.fit(inputs, targets, validation_split=0.25)
```

##### Bias regularization

```
model = Sequential([
    Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.001, l2=0.005), 
                                    bias_regularizer=tf.keras.regularizers.l1(0.005))),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['acc'])
model.fit(inputs, targets, validation_split=0.25)
```

##### Dropouts

```
model = Sequential([
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['acc'])
model.fit(inputs, targets, validation_split=0.25)
```

Required argument is `dropout_rate`. 

Each weight connection between the two dense layers is set to zero with the probability=dropout_rate. Here, that probability is `0.5`.

This is sometimes referred to as Bernoulli Dropout, because each weight is multipled by a Bernoulli random number.


When we use Dropouts, there are typically two different ways we can run the model. Training mode is above.

When we are evaluating or fitting or predicting using the model, it is the testing mode, because we stop randomly dropping out the weights.

So,

Training mode -> has dropout
Testing mode -> no dropout


### Batch Normalization layer example:

```
# Build the model

model = Sequential([
    Dense(64, input_shape=[train_data.shape[1],], activation="relu"),
    BatchNormalization(),  # <- Batch normalisation layer
    Dropout(0.5),
    BatchNormalization(),  # <- Batch normalisation layer
    Dropout(0.5),
    Dense(256, activation='relu'),
])

```

Layer with arguments:

```
model.add(tf.keras.layers.BatchNormalization(
    momentum=0.95, 
    epsilon=0.005,
    axis = -1,
    beta_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05), 
    gamma_initializer=tf.keras.initializers.Constant(value=0.9)
))
```

### Callbacks 

`tf.keras.callbacks` has a base class called `Callbacks` which all other callbacks inherit.

```
from tf.keras.callbacks import Callbacks

class my_callback(Callback): # The following methods are ones that can be overridden
    def on_train_begin(self, logs=None):
         # Do something once at the start of training.

    def on_train_batch_begin(self, batch, logs=None):
         # Do something at the start of every batch iteration. The batch number is passed as the batch argument.

    def on_epoch_end(self, epoch, logs=None):
         # Do something at the end of every epoch. The epoch number is passed as the epoch argument.


history = model.fit(X_train, y_train, epochs=5, callbacks=[my_callback()])
```

The `callbacks` argument in `model.fit()` takes a list of all callbacks.

The `history` object is callback that is automatically included into every training run when `model.fit()` is called. The action is to recall the `loss` and `metrics` values into a dictionary in its `history` attribute, i.e., `history.histroy`.


### Early stopping and patience

Early stopping - a regularization technique that monitors the performance of the network for every epoch on a held out validation set. 

Keras has a built-in callback for this.

```
from tenforflow.keras.callbacks import EarlyStopping

model = Sequential([
    ...
])

model.compile(...)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.01, mode='max')

model.fit(X_train, y_train, validation_split=0.2, epochs=100, callbacks=[early_stopping])
```

To set the metric to monitor, use the `monitor` keyword arg. The default is `val_loss`. The string passed to this is the same as one of the keys in the history object. 

The `patience` keyword arg is set to 0 by default. This means that as soon as the performance gets worse from one epoch to the next, the training is terminated. This is not ideal if the training is noisy. Setting it to 5 makes sense because training stops if for 5 consecutive epochs the performance hasn't improved.

The `min_delta` keyword arg is used to set a threshold for what counts as improvement in performance. Anything under this threshold is treated as a decrease, and the patience counter is increased by one.

The `mode` keyword arg is set to `auto`. The direction is inferred automatically, i.e., increasing or decreasing. But this can be set manually to `max` or `min` to indicate that we aim to maximize or minimize, respectively, the chosen monitoring mectric. The patience counter will then use this 'direction' to identify whether the `min_delta` threshold has been hit.


### Programming Assignment

Model Validation on Iris dataset