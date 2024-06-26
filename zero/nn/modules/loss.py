import numpy as np
from .layer import Layer


class MSE(Layer):
    """ Mean Squared Error. """
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray):
        self.y_pred = y_pred
        self.y_true = y_true
        return (1 / len(y_true)) * np.sum(np.power(y_pred - y_true, 2))
    
    def backward(self):
        return (2 / len(self.y_pred)) * np.sum(self.y_pred - self.y_true)
