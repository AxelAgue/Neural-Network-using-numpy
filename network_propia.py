import numpy as np
import random

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

a = mnist.train.images
b = mnist.train.labels
train_data = a.tolist()
train_data_labels = b.tolist()


c = mnist.test.images
d = mnist.test.labels
test_data = c.tolist()
test_data_labels = d.tolist()



class Network(object):

    def __init__(self, sizes):

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]




    def mini_batch(self, mini_batch_size, epochs, train_data_labels, train_data, eta, test_data=None, test_data_labels=None):

        #eleccion del mini batch
        for j in range(epochs):
            k = np.random.randint(0, len(train_data_labels) - mini_batch_size)
            mini_batch = train_data[k:k + mini_batch_size]
            mini_batch_labels = train_data_labels[k:k + mini_batch_size]

            self.GD_mini_batch(mini_batch, mini_batch_labels, eta, mini_batch_size)


            if test_data is not None:
                print("Epoch {0}".format(j, self.evaluate(test_data, test_data_labels), len(test_data)))




    def GD_mini_batch(self, mini_batch, mini_batch_labels, eta, mini_batch_size):


        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for i in range(mini_batch_size):
            s = mini_batch[i]
            y = mini_batch_labels[i]
            delta_nabla_b, delta_nabla_w = self.backprop(s, y)


            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]



        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]

        self.biases = [b - (eta / len(mini_batch)) * nb
                        for b, nb in zip(self.biases, nabla_b)]

        #print(nabla_b[0])

        #print(nabla_w[1])









    def backprop(self, s, y):


        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        zs = []


        activations = [s]

        a = np.dot(self.weights[0], s) + self.biases[0].reshape(1, 3)

        zs.append(a)
        z = act_func(a)
        activations.append(z)
        b = np.dot(self.weights[1], z.reshape(3, 1))
        zs.append(b)
        z1 = softmax(b)
        activations.append(z1)




        delta = (activations[-1] - y).mean() * act_func_prime(zs[-1])



        nabla_b[-1] = delta
        nabla_w[-1] = np.multiply(delta, activations[-2].reshape(1, 3))

        for l in range(2, self.num_layers):
         z = zs[-l]
         sp = act_func_prime(z)
         delta = np.dot(self.weights[-l + 1].reshape(3, 10), delta) * sp.reshape(3, 1)

         nabla_b[-l] = delta
         nabla_w[-l] = np.multiply(delta, activations[-l - 1])


        return (nabla_b, nabla_w)

    def feedforward(self, z):

        # for i in range(self.num_layers - 1):
        a = act_func(np.dot(self.weights[0], z) + self.biases[0].reshape(1, 3))
        z = np.dot(self.weights[1], a.reshape(3, 1)) + self.biases[1]

        return softmax(z)




    def evaluate(self, test_data, test_data_labels):

     c1 = [self.feedforward(x) for x in test_data]
     c2 = test_data_labels
     print(c1[5].reshape(1, 10))
     print(c2[5])

     mse = ((c1[5] - c2[5]) ** 2).mean()
     print(mse)


"""
     
     prediccion_correcta = 0

     for i in range(len(test_data)):
         mse = ((c1[i] - c2[i]) ** 2).mean()

         if mse <= 0.00:
             prediccion_correcta += 1


     return prediccion_correcta
"""




def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)



# sigmoid
def act_func(z):
    return 1.0 / (1.0 + np.exp(-z))


def act_func_prime(z):
    return act_func(z) * (1 - act_func(z))


net = Network([784, 3, 10])
net.mini_batch(30, 10000, train_data_labels, train_data, 3e15, test_data=test_data, test_data_labels=test_data_labels)














