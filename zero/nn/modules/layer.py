from abc import ABCMeta, abstractmethod


class Layer(metaclass=ABCMeta):
    """ Base layer of neural network. """
    @abstractmethod 
    def forward(self, *args, **kwargs):
        """ Forward propagation method. """
        pass

    @abstractmethod
    def backward(self, *args, **kwargs):
        """ Backward propagation method. """
        pass
