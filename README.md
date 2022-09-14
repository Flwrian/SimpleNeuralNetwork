# Simple Neural Network

A [Neural Network](https://fr.wikipedia.org/wiki/R%C3%A9seau_de_neurones_artificiels) is trained by providing data to it.

- Input data
- Expected data

![alt](https://upload.wikimedia.org/wikipedia/commons/thumb/3/3d/Neural_network.svg/1920px-Neural_network.svg.png)

Note that [Neural Network](https://fr.wikipedia.org/wiki/R%C3%A9seau_de_neurones_artificiels) is still in testing since there are missing parameters like learning rate and more parameters for customization. I might be wrong about some things, I'm currently learning how a [Neural Network](https://fr.wikipedia.org/wiki/R%C3%A9seau_de_neurones_artificiels) works.

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
print(network.predict(X))
```

> Output:
```python
[[0.01038789]
 [0.99123457]
 [0.98944122]
 [0.00863672]]
```

The interpretation of the output is simple. Each number, correspond to the accuracy.

1st Output: $(1-0.01038789) * 100$ = $0.98961211$

This means that the [Neural Network](https://fr.wikipedia.org/wiki/R%C3%A9seau_de_neurones_artificiels) is sure at ~99% that the output is **NOT** 1 here (or 1% sure that the output is 1).

2nd Output: $0.99123457 * 100$ = $0.98961211$

This means that the [Neural Network](https://fr.wikipedia.org/wiki/R%C3%A9seau_de_neurones_artificiels) is sure at ~99% that the output is 1 here (or 1% sure that the output is **not** 1).


----------


Now let's say that if a row of $X$ is **even**, we want to output 1 in $y$[0][0] (the corresponding output neuron).

We will re-use our trained [Neural Network](https://fr.wikipedia.org/wiki/R%C3%A9seau_de_neurones_artificiels) to predict our input once.

Note that we are re-using the same inputs, but in a different order.

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

# New input data
X = np.array([  
                [1,0,1], # Even (1)
                [1,1,1], # Odd (0)
                [1,0,1], # Even (1)
                [0,1,1]  # Even (1)
            ])

# Expected output data
y = np.array([  
                [1], # Should output 1
                [0], # Should output 0
                [1], # Should output 1
                [1]  # Should output 1
            ])

print(network.predict(X))
```

> Output:

```python
[[0.99061512]
 [0.0109055 ]
 [0.99061512]
 [0.99236198]]
```

This match with our expected output.

----------
Note that the more you train the [Neural Network](https://fr.wikipedia.org/wiki/R%C3%A9seau_de_neurones_artificiels), the more accurate you will get on your results. If you don't train your [Neural Network](https://fr.wikipedia.org/wiki/R%C3%A9seau_de_neurones_artificiels) enough, sometimes it can be false. You need to train the [Neural Network](https://fr.wikipedia.org/wiki/R%C3%A9seau_de_neurones_artificiels) in a way that it know a lot of "combinaison" so it can try his best to predict an output.

Example (same code, inputs and outputs):

```python
#100 iterations
[[0.52833502]
 [0.48455771]
 [0.52833502]
 [0.48459066]]
```

```python
#1000 iterations
[[0.92622703]
 [0.07174901]
 [0.92622703]
 [0.95602922]]
```

```python
#100000 iterations
[[0.99706991]
 [0.00297979]
 [0.99706991]
 [0.99707101]]
```