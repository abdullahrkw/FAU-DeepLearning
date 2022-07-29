import torch as t
from trainer import Trainer
import model
import sys
import torchvision as tv

epoch = int(sys.argv[1])
#TODO: Enter your model here
model = model.ResNet()

crit = t.nn.BCELoss()
trainer = Trainer(model, crit, cuda=False)
trainer.restore_checkpoint(epoch)
trainer.save_onnx('checkpoint_{:03d}.onnx'.format(epoch))
