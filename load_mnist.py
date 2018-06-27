import os
import struct
import numpy as np
import matplotlib.pyplot as plt

class MNIST_Loader(object):
    def load_mnist(self, path, kind='train'):
        
        labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
        images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)

        with open(labels_path, 'rb') as lbpath:
            
            magic, n = struct.unpack('>II', lbpath.read(8))
            labels = np.fromfile(lbpath, dtype=np.uint8)

        with open(images_path, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
            
            images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

        self.images = images
        self.labels = labels
        return images, labels

    def showDigits(self):
        
        fig, ax = plt.subplots(nrows=2, ncols=5, sharex = True, sharey = True,)
        ax = ax.flatten()
        for i in range(10):
            img = self.images[self.labels == i][0].reshape(28, 28)
            ax[i].imshow(img, cmap='Greys', interpolation='nearest')

        plt.tight_layout()
        plt.show()

    def showExampleDigit(self, digit):
        
        fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
        ax = ax.flatten()
        for i in range(25):
            img = self.images[self.labels == digit][i].reshape(28, 28)
            ax[i].imshow(img, cmap='Greys', interpolation='nearest')

        plt.tight_layout()
        plt.show()

    def __init__(self):
        pass


if __name__ == '__main__':
    #example usage
    a = MNIST_Loader()
    X_train, y_train = a.load_mnist('./data')
    print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))
    print (y_train)
    #a.showDigits()
    a.showExampleDigit(3)
