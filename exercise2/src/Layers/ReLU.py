from Layers.Base import BaseLayer

class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()
        
    def forward(self, input_tensor):
        input_tensor[input_tensor <= 0] = 0 #max(0, x)
        self.x = input_tensor
        return input_tensor

    def backward(self, error_tensor):
        # # if x <= 0, output is 0. if x > 0, output is 1
        # # https://stackoverflow.com/questions/32546020/neural-network-backpropagation-with-relu
        # # https://datascience.stackexchange.com/questions/19272/deep-neural-network-backpropogation-with-relu
        self.x[self.x > 0] = 1 #gradient
        return error_tensor * self.x
