import math
from NeuralNetwork import NeuralNetwork
import numpy as np

# Our problem is to predict if the sum of the input is greater than 10.
# We will use 1x3 input and 1x1 output.
# The sum of the input will be a random number between 0 and 15.
# The output will be 1 if the sum is greater than 10 and 0 if it is not.




max = 6

# Create the input data
input = np.random.randint(0, max, (1, 3))

# Create the expected output data
expected = np.array([0])
if input.sum() >= 10:
    expected = np.array([1])

# Create the neural network
nn = NeuralNetwork(input, expected)

def train(epoch):
    for i in range(epoch):
        # Create the input data
        input = np.random.randint(0, max, (1, 3))
        expected = np.array([0])
        if input.sum() >= 10:
            expected = np.array([1])

        nn.setExpected(expected)
        nn.setInputs(input)

        # Train the neural network
        nn.train(15000)
        print("Generation:",i," | input:",input," | expected:",expected," | output:",nn.predict(input),round(i/epoch*100, 2),"% complete", "Valid? ", round(nn.predict(input)[0][0]) == expected)


train(round(math.pow(max, 3)))
input = np.array([[1, 3, 4]])
nn.input = input
print("input:",input," | output:",round(nn.predict(input)[0][0], 4))

# Save the neural network
nn.saveNetwork()