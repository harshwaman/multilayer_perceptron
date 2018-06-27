import numpy as np
from scipy.special import expit
import sys
import os
from optimizers import *

def print_shape(nparry):
    print("{}".format(nparry.shape))

class NeuralNetwork(object):

    def __init__(self, n_output, n_features, n_hidden=30, l2=0.0, epochs=500,
                 learning_rate=0.001, momentum_const=0.0, decay_rate=0.0,
                 dropout=False, minibatch_size=1,
                 optimizer = 'Gradient Descent', activation = 'relu',
                 nesterov = False, check_gradients = False, early_stop = None, metrics = ['Accuracy']):
        self.n_output = n_output
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.w1, self.w2 = self.initialize_weights()
        self.l2 = l2
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.momentum_const = momentum_const
        self.decay_rate = decay_rate
        self.dropout = dropout
        self.minibatch_size = minibatch_size
        self.nesterov = nesterov
        self.check_gradients = check_gradients
        supported_optimizers = ['Gradient Descent', 'Momentum', 'Nesterov', 'Adam', 'Adagrad', 'Adadelta', 'RMSProp']
        if optimizer not in supported_optimizers:
            print("Error: unsupported optimizer requested.")
            print("Available optimizers: {}".format(supported_optimizers))
            exit()
        else:
            self.optimizer = optimizer
        supported_activations = ['relu', 'tanh', 'sigmoid', 'maxout', 'elu']
        if activation not in supported_activations:
            print("Error: unsupported activation requested.")
            print("Available activations: {}".format(supported_activations))
        else:
            self.activation = activation
        self.early_stop = early_stop
        SUPPORTED_METRICS = ['Accuracy', 'Precision', 'Recall', 'AUC']
        for elem in metrics:
            assert elem in SUPPORTED_METRICS
        self.metrics = metrics


    def initialize_weights(self):
        
        w1 = np.random.uniform(-1.0, 1.0, size = self.n_hidden * (self.n_features + 1)).reshape(self.n_hidden, (self.n_features + 1))/(self.n_features + 1)
        w2 = np.random.uniform(-1.0, 1.0, size=self.n_output*(self.n_hidden+1)).reshape(self.n_output, self.n_hidden+ 1)/(self.n_hidden + 1)
        return w1, w2

    def encode_labels(self, y, num_labels):
        
        onehot = np.zeros((num_labels, y.shape[0]))
        for i in range(y.shape[0]):
            onehot[y[i], i] = 1.0
        return onehot

    def softmax(self, v):
         
        logC = -np.max(v)
        return np.exp(v + logC)/np.sum(np.exp(v + logC), axis = 0)

    def tanh(self, z, deriv=False):
        
        return np.tanh(z) if not deriv else 1 - np.square(np.tanh(z))

    def relu(self, z, deriv = False):
        if not deriv:
            relud = z
            relud[relud < 0] = 0
            return relud
        deriv = z
        deriv[deriv <= 0] = 0
        deriv[deriv > 0] = 1
        return deriv


    def add_bias_unit(self, X, column=True):
        
        if column:
            bias_added = np.ones((X.shape[0], X.shape[1] + 1))
            bias_added[:, 1:] = X
        else:
            bias_added = np.ones((X.shape[0] + 1, X.shape[1]))
            bias_added[1:, :] = X

        return bias_added

    def compute_dropout(self, activations, dropout_prob = 0.5):
        
        if dropout_prob < 0 or dropout_prob > 1:
            dropout_prob = 0.5
        
        activations/=dropout_prob    
        mult = np.random.binomial(1, 0.5, size = activations.shape)
        activations*=mult
        return activations

    def forward(self, X, w1, w2, do_dropout = True):
        
        a1 = self.add_bias_unit(X)
        if self.dropout and do_dropout: a1 = self.compute_dropout(a1)
        
        z2 = w1.dot(a1.T)

        a2 = self.tanh(z2)
        
        a2 = self.add_bias_unit(a2, column=False)
        if self.dropout and do_dropout: a2 = self.compute_dropout(a2)
        
        z3 = w2.dot(a2)
        
        a3 = self.softmax(z3)
        return a1, z2, a2, z3, a3

    def get_cost(self, y_enc, output, w1, w2):
        
        cost = - np.sum(y_enc*np.log(output))
        
        l2_term = (self.l2/2.0) * (np.sum(np.square(w1[:, 1:])) + np.sum(np.square(w2[:, 1:])))
        cost = cost + l2_term
        return cost/y_enc.shape[1]

    def backprop(self, a1, a2, a3, z2, y_enc, w1, w2):
        
        sigma3 = a3 - y_enc
        z2 = self.add_bias_unit(z2, column=False)
        sigma2 = w2.T.dot(sigma3) * self.tanh(z2, deriv=True)
        
        sigma2 = sigma2[1:, :]
        grad1 = sigma2.dot(a1)
        grad2 = sigma3.dot(a2.T)
         
        grad1[:, 1:]+= (w1[:, 1:]*self.l2) # derivative of .5*l2*w1^2
        grad2[:, 1:]+= (w2[:, 1:]*self.l2) # derivative of .5*l2*w2^2
        return grad1, grad2

    def accuracy(self, X_train, y_train):
        
        y_train_pred = self.predict(X_train)
        diffs = y_train_pred - y_train
        count = 0.
        for i in range(y_train.shape[0]):
            if diffs[i] != 0:
                count+=1
        return 100 - count*100/y_train.shape[0]



    def predict(self, X, dropout = False):
        
        a1, z2, a2, z3, a3 = self.forward(X, self.w1, self.w2, do_dropout = False)
        
        y_pred = np.argmax(a3, axis = 0)
        return y_pred



    def fit(self, X, y, print_progress=True):
        
        X_data, y_data = X.copy(), y.copy()
        y_enc = self.encode_labels(y, self.n_output)
        
        prev_grad_w1 = np.zeros(self.w1.shape)
        prev_grad_w2 = np.zeros(self.w2.shape)
        print("fitting")
        costs = []
        grad_1_li, grad_2_li = [], [] 

        
        for i in range(self.epochs):
            previous_accuracies = []
            self.learning_rate /= (1 + self.decay_rate*i)
            mini = np.array_split(range(y_data.shape[0]), self.minibatch_size)
            grads_w1, grads_w2 = [], [] # needed if we want to remember averages of gradients across time
            for idx in mini:
                #feed feedforward
                a1, z2, a2, z3, a3= self.forward(X_data[idx], self.w1, self.w2)
                cost = self.get_cost(y_enc=y_enc[:, idx], output=a3, w1=self.w1, w2=self.w2)
                costs.append(cost)

                #compute gradient via backpropagation

                grad1, grad2 = self.backprop(a1=a1, a2=a2, a3=a3, z2=z2, y_enc=y_enc[:, idx], w1=self.w1, w2=self.w2)
                grad_1_li.append(grad1)
                grad_2_li.append(grad2)

                if self.check_gradients:
                    eps = 10e-4
                    # compute numerical gradient
                    w1_check, w2_check = self.w1, self.w2
                    g1_check, g2_check = grad1, grad2
                    w1_check = w1_check.reshape((w1_check.shape[0] * w1_check.shape[1]))
                    w2_check = w2_check.reshape((w2_check.shape[0] * w2_check.shape[1]))
                    g1_check = g1_check.reshape((g1_check.shape[0] * g1_check.shape[1]))
                    g2_check = g2_check.reshape((g2_check.shape[0] * g2_check.shape[1]))
                    for check in [w1_check, w2_check, g1_check, g2_check]:
                        print_shape(check)
                    for i in range(w1_check.shape[0]):
                        E = np.zeros(w1_check.shape[0])
                        E[i] = 1
                        w1_check_2 = w1_check + (eps * E)
                        w1_check_3 = w1_check - (eps * E)
                        for j in range(w1_check.shape[0]):
                            if j != i:
                                assert w1_check_2[j] == w1_check[j], "Houston we have a problem"
                                assert w1_check_3[j] == w1_check[j], "Houston we have a problem"
                        J1 = self.get_cost(y_enc=y_enc[:, idx], output=a3, w1=w1_check_2.reshape((self.w1.shape[0], self.w1.shape[1])), w2=self.w2)
                        J2 = self.get_cost(y_enc=y_enc[:, idx], output=a3, w1=w1_check_3.reshape((self.w1.shape[0], self.w1.shape[1])), w2=self.w2)
                        numerical_gradient = (J1 - J2) / (2 * eps)
                        if i % 100 == 0:
                            print("numerical gradient is {} and actual gradient is {}".format(numerical_gradient, g1_check[i]))

                
                w1_update, w2_update = self.learning_rate*grad1, self.learning_rate*grad2
                
                if self.nesterov:
                    
                    v1 = self.momentum_const * prev_grad_w1 - w1_update
                    v2 = self.momentum_const * prev_grad_w2 - w2_update
                    self.w1 += -self.momentum_const * prev_grad_w1 + (1 + self.momentum_const) * v1
                    self.w2 += -self.momentum_const * prev_grad_w2 + (1 + self.momentum_const) * v2
                else:
                    # gradient update: w += -alpha * gradient.
                    # use momentum - add in previous gradient mutliplied by a momentum hyperparameter.
                    self.w1 += -(w1_update + (self.momentum_const*prev_grad_w1))
                    self.w2 += -(w2_update + (self.momentum_const*prev_grad_w2))
                
                prev_grad_w1, prev_grad_w2 = w1_update, w2_update

            if print_progress and (i+1) % 1 == 0:
                print("Epoch: {}".format(i + 1))
                print("Loss: {}".format(cost))
                if self.check_gradients:
                    print("Gradient Error: {}".format(w1_grad_error))
                grad_1_mag, grad_2_mag = np.linalg.norm(grad_1_li), np.linalg.norm(grad_2_li)
                acc = self.accuracy(X, y)
                previous_accuracies.append(acc)
                if self.early_stop is not None and len(previous_accuracies) > 3:
                    if abs(previous_accuracies[-1] - previous_accuracies[-2]) < self.early_stop and abs(previous_accuracies[-1] - previous_accuracies[-3]) < self.early_stop:
                        print("Early stopping, accuracy has stayed roughly constant over last 100 iterations.")
                        break

                print("Training Accuracy: {}".format(acc))

        return self
