import numpy as np
from NeuralNetwork import NeuralNetwork
from load_mnist import MNIST_Loader
import _pickle as pickle

if __name__ == '__main__':
    a = MNIST_Loader()
    X_train, y_train = a.load_mnist('./data')
    X_test, y_test = a.load_mnist('./data', 't10k')
    print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))
    nn = NeuralNetwork(n_output=10,
                  n_features=X_train.shape[1],
                  n_hidden=50,
                  l2=0.1,
                  epochs=100,
                  minibatch_size=50,)
    nn.fit(X_train, y_train, print_progress=True)

    filename = 'finalized_model.sav'
    pickle.dump(nn, open(filename, 'wb'))
