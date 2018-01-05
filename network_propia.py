import numpy as np
import random

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

train_data = mnist.train.images
train_data_labels = mnist.train.labels

test_data = mnist.test.images
test_data_labels = mnist.test.labels


class Network(object):
    def __init__(self, sizes):

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]



    def feedfoward(self, z):

        for i in range(self.num_layers - 1):
            z = act_func(np.sum(np.multiply(self.weights[i], z) + self.biases[i], axis=1))

        return softmax(z)




    def evaluate(self, test_data, test_data_labels):

     c1 = [self.feedfoward(x) for x in test_data]
     c2 = test_data_labels

     prediccion_correcta = 0

     for i in xrange(len(test_data)):
         mse = ((c1[i] - c2[i]) ** 2).mean()


         if mse <= 0:
             prediccion_correcta += 1
     return prediccion_correcta



    def mini_batch(self, mini_batch_size, epochs, train_data_labels, train_data, eta):

        #eleccion del mini batch
        for i in range(epochs):
            k = np.random.randint(0, len(train_data_labels) - mini_batch_size)
            mini_batch = [train_data[k:k + mini_batch_size]]
            mini_batch_labels = [train_data_labels[k:k + mini_batch_size]]


            self.GD_mini_batch(mini_batch, mini_batch_labels, eta, mini_batch_size)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data, test_data_labels), len(train_data)))
            else:
                print("Epoch {0} complete".format(j))



    def GD_mini_batch(self, mini_batch, mini_batch_labels, eta, mini_batch_size):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for i in range(len(mini_batch)):
            delta_nabla_b, delta_nabla_w = self.backprop(mini_batch, mini_batch_labels, mini_batch_size)


            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]


        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]




    def backprop(self, mini_batch, mini_batch_labels, mini_batch_size):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        zs = []
        activations = []

        for j in range(mini_batch_size):
            s = mini_batch[j]
            for i in range(self.num_layers - 1):
                s = np.sum(np.multiply(self.weights[i], s) + self.biases[i], axis=1)
                zs.append(s)
                z = act_func(s)
                activations.append(z)
            return zs, activations
            delta = (activations[-1] - mini_batch_labels[j]).mean() * act_func_prime(zs[-1])
            nabla_b[-1] = delta
            nabla_w[-1] = np.dot(delta, activations[-2].transpose())
            for l in range(2, self.num_layers):
             z = zs[-l]
             sp = act_func_prime(z)
             delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
             nabla_b[-l] = delta
             nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
            return (nabla_b, nabla_w)




def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)



# sigmoid
def act_func(z):
    return 1.0 / (1.0 + np.exp(-z))


def act_func_prime(z):
    return act_func(z) * (1 - act_func(z))


net = Network([784, 3, 10])
net.mini_batch(10, 30, train_data_labels, train_data, 0.1)
















