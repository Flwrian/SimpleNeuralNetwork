# Simple Neural Network

A [Neural Network](https://fr.wikipedia.org/wiki/R%C3%A9seau_de_neurones_artificiels) is trained by providing data to it.

- Input data
- Expected data

![alt](https://upload.wikimedia.org/wikipedia/commons/thumb/3/3d/Neural_network.svg/1920px-Neural_network.svg.png)

## Example:
```python
# Input data
X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1]  ])

# Expected output data
y = np.array([  [0],
                [1],
                [1],
                [0]  ])

network = NeuralNetwork(X, y)
network.train(10000)
print(network.process(X))
```

> Output:
```python
[[0.01038789]
 [0.99123457]
 [0.98944122]
 [0.00863672]]
```

