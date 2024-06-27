import numpy as np
from .layer import Layer


class ReLU(Layer):
    def __init__(self):
        self.input_matrix: np.ndarray
        
    def forward(self, input_matrix: np.ndarray):
        self.input_matrix = input_matrix
        return np.maximum(0, input_matrix)
    
    def backward(self, output_gradient: np.ndarray, learning_rate: float):
        return learning_rate * output_gradient * (self.input_matrix >= 0)
