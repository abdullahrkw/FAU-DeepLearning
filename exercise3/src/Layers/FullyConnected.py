import numpy as np

from Layers.Base import BaseLayer


class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.trainable=True
        self._optimizer=None
        #because we add Bias into the input tensor so it has to be (p,m) ,(m,q) -: in this case (9,5),(5,3) (# TestCase)
        self.weights=np.random.uniform(low=0.0, high=1.0, size=(input_size +  1, output_size))

    def forward(self,input_tensor):
        batch_size = input_tensor.shape[0]
        self.bias = np.ones((batch_size, 1))
        self.input_tensor = np.hstack((input_tensor, self.bias))
        result = np.dot(self.input_tensor, self.weights)
        return  result

    def initialize(self, weight_initializer, bias_initializer):
        self.weights = weight_initializer.initialize((self.input_size +  1, self.output_size), self.input_size, self.output_size)

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer=optimizer

    def backward(self,error_tensor):
        weights = np.delete(self.weights, self.weights.shape[0] - 1, axis=0)
        self.error_tensor = np.dot(error_tensor, weights.T)
        self.gradient_weights = np.dot(self.input_tensor.T, error_tensor)
        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)

        return self.error_tensor
