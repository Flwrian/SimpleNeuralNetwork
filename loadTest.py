from NeuralNetwork import NeuralNetwork
import numpy as np

input = np.array([[1, 3, 4]])
expected = np.array([[0]])
nn = NeuralNetwork(input, expected)

nn.loadNetwork()


print("input:",input," | output:",round(nn.predict(input)[0][0], 4))