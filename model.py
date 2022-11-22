
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.model = nn.Sequential()
        enc_channels = args.enc_channels
        
        for i, pair in enumerate(enc_channels):
            self.model.add_module(f"conv_{i}" , ConvBlock(*pair))
            
    def forward(self, x):
        return self.model(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels =out_channels,kernel_size=3,stride=1,padding=1)
        self.pool = nn.MaxPool2d((2,2))
        self.activation = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(in_channels=out_channels,out_channels =out_channels,kernel_size=3,stride=1,padding=1)
        self.activation1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        
    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.pool(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        return x

class Decoder(nn.Module):
    def __init__(self,args) :
        super(Decoder, self).__init__()
        self.model = nn.Sequential()
        attn_resolutions = [16,32]
        resolution = 2
        dec_channels = args.dec_channels
        for i, pair in enumerate(dec_channels):
            self.model.add_module(f"deconv_{i}" , Upsampling(*pair))

    def forward(self, x):
        return self.model(x)

        

class Upsampling(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Upsampling, self).__init__()
        self.up = nn.Upsample(scale_factor=2,mode='nearest')
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1)
        self.activation = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1)
        self.activation1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(out_channels)
    def forward(self, inputs):
        x = self.up(inputs)
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.conv1(x)
        x = self.bn1(x)
        return self.activation1(x)



if __name__ == "__main__":
    encoder_channels = [(3,8), (8,16), (16,32), (32,64),\
             (64, 128), (128,128), (128,256), (256,512)]
    decoder_channels = [(512,256),(256,128),(128,128),\
        (128,64),(64,32),(32,16),(16,8),(8,3)]

  

    encoder = Encoder(encoder_channels)
    X = np.random.rand(2, 3, 512, 512)
    x = encoder(torch.from_numpy(X).float())


