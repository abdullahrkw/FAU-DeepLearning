import copy
import numpy as np
from Layers import Base
from Layers.Helpers import compute_bn_gradients


class BatchNormalization(Base.BaseLayer):
    def __init__(self, channels) -> None:
        super().__init__()
        self.trainable = True
        self.channels = channels
        self.mean_training_agg = None
        self.var_training_agg = None
        self._optimizer = None
        self.bias = None
        self.weights = None
        self.decay = 0.8
        self.initialize()

    def initialize(self):
        self.weights = np.ones(self.channels)
        self.bias = np.zeros(self.channels)        

    def forward(self, inpT):
        self.inpT = inpT
        if len(self.inpT.shape) == 4:
            inpT = self.reformat(inpT)
        self.meanB = np.mean(inpT, axis=0)
        self.varB = np.var(inpT,axis=0)
        if self.mean_training_agg is None:
            self.mean_training_agg = self.meanB
            self.var_training_agg = self.varB
        if self.testing_phase:
            norm_inpT = (inpT - self.mean_training_agg)/np.sqrt(self.var_training_agg + np.finfo(float).tiny)
        else:
            norm_inpT = (inpT - self.meanB)/np.sqrt(self.varB + np.finfo(float).tiny)
            self.mean_training_agg = self.decay*self.mean_training_agg + (1 - self.decay)*self.meanB
            self.var_training_agg = self.decay*self.var_training_agg + (1 - self.decay)*self.varB
        self.norm_inpT = norm_inpT
        output = self.weights*norm_inpT + self.bias
        if len(self.inpT.shape) == 4:
            output = self.reformat(output)
        return output


    def backward(self, errT):
        self.errT = errT
        if len(self.errT.shape) == 4:
            errT = self.reformat(errT)
        inpT = self.inpT
        if len(self.inpT.shape) == 4:
            inpT = self.reformat(self.inpT)
        self.gradient_weights = np.sum(errT*self.norm_inpT, axis=0)
        self.gradient_bias = np.sum(errT, 0)
        self.gradient_input =  compute_bn_gradients(errT, inpT, self.weights, self.meanB, self.varB)
        if self.optimizer is not None:
            self.weights = copy.deepcopy(self.optimizer).calculate_update(self.weights, self.gradient_weights)
            self.bias = copy.deepcopy(self.optimizer).calculate_update(self.bias, self.gradient_bias)
        if len(self.errT.shape) == 4:
            self.gradient_input = self.reformat(self.gradient_input)
        return self.gradient_input

    def reformat(self, inpT):
        if len(inpT.shape) == 4:
            B,H,M,N = inpT.shape
            two_d = np.reshape(inpT, (B, H, M*N)).transpose(0,2,1).reshape((B*M*N, H))
            return two_d
        elif len(inpT.shape) == 2:
            BMN, H = inpT.shape
            B = self.inpT.shape[0]
            MN = int(BMN / B)
            four_d = np.reshape(inpT, (B, MN, H)).transpose(0, 2, 1).reshape(self.inpT.shape)
            return four_d


    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer