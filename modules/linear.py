import numpy as np
from .layer import Layer


class Linear(Layer):
    """ Linear layer of neural network. 
    Performs linear transformation: wX + b. """
    def __init__(self, input_dim: int, output_dim: int):
        self.weights = np.random.randn(output_dim, input_dim)
        self.bias = np.random.randn(output_dim, 1)

    def forward(self, x: np.ndarray):
        """ Forward propagation method. """
        self.x = x
        output = self.weights @ x
        output += self.bias
        return output
    
    def backward(self, output_gradient: np.ndarray, lr: float):
        """ Back propagation method. """
        self.weights -= lr * output_gradient @ self.x.T
        self.bias -= lr * output_gradient
        output_gradient = self.weights.T @ output_gradient
        return output_gradient
