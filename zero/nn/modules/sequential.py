import numpy as np
from .layer import Layer


class Sequential(Layer):
    """ Sequential layer that connects other modules together and 
    can be used as a standalone model. """
    def __init__(self, *modules: tuple[Layer]):
        self.modules = []
        for module in modules:
            self.modules.append(module)

    def forward(self, input_matrix: np.ndarray):
        output = input_matrix
        for module in self.modules:
            output = module(output)

        return output
    
    def backward(self, output_gradient: np.ndarray, learning_rate: float):
        for module in reversed(self.modules):
            output_gradient = module.backward(output_gradient, learning_rate)

        return output_gradient
