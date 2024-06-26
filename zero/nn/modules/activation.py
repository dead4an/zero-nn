import numpy as np
from .layer import Layer


class ReLU(Layer):
    def forward(self, x: np.ndarray):
        self.x = x
        return max(0, x)
    
    def backward(self, output_gradient: np.ndarray):
        return output_gradient * (self.x >= 0)
