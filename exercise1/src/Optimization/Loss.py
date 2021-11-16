import numpy as np 

class CrossEntropyLoss(object):
    def __init__(self):
        pass
 
    def forward(self, pred, label):
        print(label)
        print(pred)
        loss = -np.sum(label*np.log(pred + 1e-15))
        return loss/pred.shape[0]
    
    def backward(self, label):
        return label
