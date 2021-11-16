import numpy as np

from Layers.Base import BaseLayer


class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super(FullyConnected, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.trainable=True

        #it has to be declare here, otherwise it throws an error for the test case (Test_gradient)
        self.optimizer=None

        #because we add Baise into the input tensor so it has to be (p,m) ,(m,q) -: in this case (9,5),(5,3) (# TestCase)
        self.weights=np.random.uniform(size=(input_size+1,output_size))

    def forward(self,input_tensor):
        # https://math.stackexchange.com/questions/1866757/not-understanding-derivative-of-a-matrix-matrix-product

        #mentioned in Des.pdf

        batch_size = np.shape(input_tensor)[0]
        biases = np.ones((batch_size, 1))

        #it should be self.input_tensor because we need to access this in backward
        self.input_tensor = np.hstack((input_tensor, biases))
        result=np.dot(self.input_tensor, self.weights)
        return  result

    #https: // stackoverflow.com / questions / 2627002 / whats - the - pythonic - way - to - use - getters - and -setters
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self,optimizer):
        self._optimizer=optimizer

    def backward(self,error_tensor):
        #to convert weight (5,3 ) to (4,3) we need to delete the last row which is for the Bias
        weights = np.delete(self.weights, np.shape(self.weights)[0] - 1,axis=0)

        self.error_tensor = np.dot(error_tensor, weights.T)
        self.gradient_weights = np.dot(self.input_tensor.T, error_tensor)

        if (self.optimizer is not None):
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
        return self.error_tensor
