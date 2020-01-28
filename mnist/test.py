from Mnist import Mnist
from NN import NN
import numpy as np

dataset = Mnist("train-images.idx3-ubyte","train-labels.idx1-ubyte")
X_train = dataset.get_images((0,20000))
y_train = dataset.get_labels((0,20000))

X_val = dataset.get_images((20001,21001))
y_val = dataset.get_labels((20001,21001))

network = NN()
network.train(X_train, y_train, iterations=10000,batch_size=1000)


print(np.mean(network.predict(X_val) == y_val))
