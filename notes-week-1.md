
### Tensorflow overview

TF uses symbolic mathematics (instead of purely numerical computations) to perform operations like automatic differentiation on a computational graph (like an NN). Another ability - perform computations on GPU.

TensorFlow is one of the most popular libraries available for this purpose; other similar libraries include PyTorch, Chainer, Apache MXNet, Caffe and Microsoft CNTK.

TensorFlow was developed by Google Brain and version 1.0.0 was released in 2017. It emerged from an earlier proprietary framework called DistBelief.

The latest release of TensorFlow 2 makes use of the Keras API as the default high-level abstraction to easily construct and customize neural networks.

Another fundamental feature of TensorFlow is its ability to develop and deploy models in multiple platforms and environments. The TensorFlow ecosystem supports development in *Python, JavaScript or Swift*, with data preprocessing and model building and training pipelines. If you want to run your model in a web browser you can use *TensorFlow.js*, or to deploy on mobile and embedded devices then you can choose *TensorFlow Lite*. And if you are interested in large-scale production environments your choice is *TensorFlow Extended*. 

There are now indeed several chipsets whose aim is solely to execute neural networks, such as the TPU from Google or the FSD chip from Tesla.

Artificial General Intelligence: https://www.forbes.com/sites/cognitiveworld/2019/06/10/how-far-are-we-from-achieving-artificial-general-intelligence/#3e9e93b76dc4


### Tensorflow 2 updates from 1

- First setup variables, loss function, optimizers before training (`tf.get_variable`). These would all form a computational graph.
- Also had to define placeholders which are entrypoints to the graph (`tf.placeholder`)


TF2:
- Eager exec is default => No need to define variables (`tf.Variable`). Dynamically pass static objects instead of a variable
- Keras has become default high-level API. Bundled into TF2
- API cleanup -> optimized, better docs, better organization of the code in TF2 


*Laurence Moroney* -> 1992 - England was in recession - one scheme to get out of recession was AI. Found 20 people with college degrees but no jobs - spearhead a process to make AI the norm. Firebase predictions - for website analytics - uses TF as the base. 

Google Colab uses TF1 by default. Need to update it and restart the runtime.
Colab can also load jupyter notebooks from your local machine and Github.

Webapp to easily convert TF1 to TF2: http://tf2up.ml




