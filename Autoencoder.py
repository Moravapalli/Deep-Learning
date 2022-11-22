
import torch
import torch.nn as nn
from model import Encoder,Decoder

class Model(nn.Module):
    def __init__(self,args) -> None:
        super(Model,self).__init__()
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)

    def forward(self,image):
        x = self.encoder(image)
        x = self.decoder(x)
        return x



