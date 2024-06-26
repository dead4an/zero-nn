import numpy as np
from .layer import Layer


class Linear(Layer):
    """ Linear layer of neural network. 
    Performs linear transformation: wX + b. """
    def __init__(self, input_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = np.random.randn(output_dim, input_dim)
        self.bias = np.random.randn(output_dim, 1)
        self.input_matrix: np.ndarray

    def forward(self, input_matrix: np.ndarray):
        """ Forward propagation method. """
        self.input_matrix = input_matrix
        output = self.weights @ input_matrix
        output += self.bias
        return output

    def backward(self, output_gradient: np.ndarray, learning_rate: float):
        """ Back propagation method. """
        # Update self parameters
        weights_gradient = output_gradient @ self.input_matrix.T
        self.weights -= learning_rate * weights_gradient

        # NOTE: check bias update or replace with better
        self.bias = output_gradient.mean(axis=1).reshape(self.output_dim, 1)

        # Pass output gradient to previous layer
        output_gradient = self.weights.T @ output_gradient
        return output_gradient
