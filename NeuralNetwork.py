import numpy as np

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

# Upgrade possible:
# Refactor? -> Layer -> Neuron
# aJOUT DES BIAS

class NeuralNetwork:
    """ Very Simple Neural Network with 1 hidden layer.\n
    You can use this class to train a neural network and then use the weights to process data.There are 3 layers in the network:\n
    1. Input layer\n
    2. Hidden layer\n
    3. Output layer\n
    
    Example data:\n
    input = np.array([  [0,0,1],\n
                    [0,1,1],\n
                    [1,0,1],\n
                    [1,1,1]  ])\n

    expected = np.array([  [0],\n
                    [1],\n
                    [1],\n
                    [0]  ])\n

    You can tweak the input and expected data to train the network to do different things but the input and expected data must be the same size: -> look at shapes for more info\n
    """

    def __init__(self, input, expected):
        self.syn0 = 2*np.random.random((3,4)) - 1
        self.syn1 = 2*np.random.random((4,1)) - 1
        self.epoch = 0

        # Input data
        self.input = input

        # Expected output data
        self.expected = expected

    def setWeights(self, weights):
        self.syn0 = weights[0]
        self.syn1 = weights[1]

    def predict(self, input):
        """ Process the input data through the neural network and return the predicted output. """""

        # Feed forward through layers 0, 1, and 2
        l0 = input
        l1 = nonlin(np.dot(l0,self.syn0))
        l2 = nonlin(np.dot(l1,self.syn1))

        # How much did we miss the target value?
        l2_error = self.expected - l2

        # In what direction is the target value?
        # were we really sure? If so, don't change too much.
        l2_delta = l2_error*nonlin(l2,deriv=True)

        # How much did each l1 value contribute to the l2 error (according to the weights)?
        l1_error = l2_delta.dot(self.syn1.T)

        # In what direction is the target l1?
        # were we really sure? If so, don't change too much.
        l1_delta = l1_error * nonlin(l1,deriv=True)

        self.syn1 += l1.T.dot(l2_delta)
        self.syn0 += l0.T.dot(l1_delta)

        self.epoch += 1

        return l2

    def train(self, epochs=0):
        """ Train the neural network with number of epochs. """""
        for i in range(epochs):
            self.predict(self.input)