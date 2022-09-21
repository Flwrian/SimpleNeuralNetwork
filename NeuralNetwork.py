from audioop import bias
import numpy as np
import random
# Sigmoid function (activation function)
def nonlin(x, deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

# Relu function (activation function)
# Not implemented yet
def relu(x, deriv=False):
    if(deriv==True):
        return 1. * (x > 0)
    return x * (x > 0)

random.seed(1)

class NeuralNetwork:
    """ Very Simple Neural Network with 1 hidden layer.\n
    You can use this class to train a neural network and then use the weights to process data.There are 3 layers in the network:\n
    1. Input layer\n
    2. Hidden layer\n
    3. Output layer\n

    You can tweak the input and expected data to train the network to do different things but the input and expected data must be the same size: -> look at shapes for more info\n
    """

    def __init__(self, input, expected):

        # syn0 = 2x3 matrix of weights (3x2 matrix)
        self.syn0 = 2*np.random.random((3,4)) - 1

        # syn1 = bias (1x1 matrix)
        self.syn1 = 2*np.random.random((4,1)) - 1

        # Epoch (number of times the neural network has been trained)
        self.epoch = 0

        # Input data
        self.input = input

        # Expected output data
        self.expected = expected

        self.loadNetwork()

    def setWeight(self, weights):
        """ Set the weights. """""
        self.syn0 = weights

    def setBias(self, bias):
        """ Set the bias. """""
        self.syn1 = bias

    def predict(self, input):
        """ Process the input data through the neural network and return the predicted output. """""

        # Feed forward through layers 0, 1, and 2
        l0 = input
        l1 = nonlin(np.dot(l0, self.syn0))
        l2 = nonlin(np.dot(l1, self.syn1))

        # Back propagation of errors using the chain rule.
        l2_error = self.expected - l2
        l2_delta = l2_error * nonlin(l2, deriv=True)

        l1_error = l2_delta.dot(self.syn1.T)
        l1_delta = l1_error * nonlin(l1, deriv=True)

        # Update weights
        self.syn1 += l1.T.dot(l2_delta)
        self.syn0 += l0.T.dot(l1_delta)

        self.epoch += 1

        return l2

    def train(self, epochs):
        """ Train the neural network with number of epochs. """""
        for i in range(epochs):
            self.predict(self.input)

    # Save the weights
    def saveWeights(self):
        np.save("weights.npy", self.syn0)

    # Save the bias
    def saveBias(self):
        np.save("bias.npy", self.syn1)

    def saveNetwork(self):
        """ Save the weights and bias to the files. The files will be named weights.npy and bias.npy. """""
        self.saveWeights()
        self.saveBias()

    # Load the weights
    def loadWeights(self):
        self.syn0 = np.load("weights.npy")

    # Load the bias
    def loadBias(self):
        self.syn1 = np.load("bias.npy")

    def loadNetwork(self):
        """ Load the weights and bias from the files. The files must be named weights.npy and bias.npy. """""
        try:
            self.loadWeights()
            self.loadBias()
            print("Network loaded successfully.")
        except:
            print("Error: Could not load the weights and bias from the files.")