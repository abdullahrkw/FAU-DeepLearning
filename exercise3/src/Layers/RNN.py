import copy
import numpy as np

from Layers.Base import BaseLayer
from Layers.FullyConnected import FullyConnected
from Layers.TanH import TanH
from Layers.Sigmoid import Sigmoid


class RNN(BaseLayer):
    def __init__(self, input_sz, hidden_sz, output_sz):
        super().__init__()
        self.trainable = True
        self.input_sz = input_sz
        self.output_sz = output_sz
        self.hidden_sz = hidden_sz
        self.hidden = np.zeros((1,self.hidden_sz))
        self._in_memory = True
        self.FC1 = FullyConnected(input_sz + hidden_sz, hidden_sz)
        self.FC1_actv = TanH()
        self.FC2 = FullyConnected(hidden_sz, output_sz)
        self.FC2_actv = Sigmoid()
        self._optimizer = None
        self._weights = None
        self._gradient_weights = None

        
    def forward(self, inpT):
        batch_size, _ = inpT.shape
        self.batch_fc2_actv =  np.zeros((batch_size, self.output_sz))
        self.batch_fc2_inp =  np.zeros((batch_size, self.hidden_sz))
        self.batch_fc1_actv = np.zeros((batch_size, self.hidden_sz))
        self.batch_fc1_inp = np.zeros((batch_size, self.input_sz + self.hidden_sz))
        fc1_input = np.zeros((1, self.input_sz + self.hidden_sz))            
        for item in range(batch_size):
            if not self.memorize:
                self.hidden = np.zeros((1,self.hidden_sz))
            fc1_input = np.hstack((inpT[item].reshape((1, self.input_sz)), self.hidden))
            self.batch_fc1_inp[item] = fc1_input
            self.hidden = self.FC1_actv.forward(self.FC1.forward(fc1_input))
            self.batch_fc1_actv[item] = np.expand_dims(self.hidden, axis=0)
            self.batch_fc2_inp[item] = np.expand_dims(self.hidden, axis=0)
            self.batch_fc2_actv[item] = self.FC2_actv.forward(self.FC2.forward(self.hidden))
        return self.batch_fc2_actv

    def backward(self, errT: np.ndarray):
        batch_size, _ = errT.shape
        batch_derv_inp = np.zeros((batch_size, self.input_sz))
        derv_hidden = np.zeros((1, self.hidden_sz))
        for item in reversed(range(batch_size)):
            derv_FC2 = self.FC2.backward(self.FC2_actv.backward(errT[item]))
            derv_FC1 = self.FC1.backward(self.FC1_actv.backward(derv_FC2 + derv_hidden))
            batch_derv_inp[item] = derv_FC1[:, :self.input_sz]
            derv_hidden = derv_FC1[:, self.input_sz:]
            self.gradient_weights += self.gradient_weights
        if self.optimizer is not None:
            self.weights = copy.deepcopy(self.optimizer).calculate_update(self.weights, self.gradient_weights)
        return batch_derv_inp

    @property
    def memorize(self):
        return self._in_memory

    @memorize.setter
    def memorize(self, value):
        self._in_memory = value

    @property
    def weights(self):
        return self.FC1.weights

    @weights.setter
    def weights(self, value):
        self.FC1.weights = value

    @property
    def gradient_weights(self):
        return self.FC1.gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, grad):
        self.FC1.gradient_weights = grad

    def initialize(self, weight_initializer, bias_initializer):
        self.FC1.initialize(weight_initializer, bias_initializer)        
        self.FC2.initialize(weight_initializer, bias_initializer)  

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer=optimizer 

    def calculate_regularization_loss(self):
        return self.optimizer.regularizer.norm(self.weights)
