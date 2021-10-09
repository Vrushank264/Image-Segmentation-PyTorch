import torch
import torch.nn as nn
import numpy as np
from torchsummary import summary

def double_conv(in_c, out_c):
    
    block = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size = 3, bias = False),
                          nn.BatchNorm2d(out_c),
                          nn.ReLU(inplace = True),
                          nn.Conv2d(out_c, out_c, kernel_size = 3, bias = False),
                          nn.BatchNorm2d(out_c),
                          nn.ReLU(inplace = True)
                          )
    return block


def crop(input, target):
    
    input_size = input.size()[2]
    target_size = target.size()[2]
    
    if input_size % 2 != 0:
        
        alpha = int(np.ceil((input_size - target_size) / 2))
        beta = int((input_size - target_size) / 2)
        return input[:, :, beta:input_size-alpha, beta:input_size-alpha]
    
    delta = (input_size - target_size) // 2
    return input[:, :, delta:input_size-delta, delta:input_size-delta]
    

class UNet(nn.Module):
    
    def __init__(self, num_classes):
        
        super(UNet, self).__init__()
        self.num_classes = num_classes
        self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        #Encoder
        self.down_conv1 = double_conv(in_c = 1, out_c = 64)
        self.down_conv2 = double_conv(in_c = 64, out_c = 128)
        self.down_conv3 = double_conv(in_c = 128, out_c = 256)
        self.down_conv4 = double_conv(in_c = 256, out_c = 512)
        self.down_conv5 = double_conv(in_c = 512, out_c = 1024)
        
        #Decoder
        self.tconv1 = nn.ConvTranspose2d(in_channels = 1024, out_channels = 512, kernel_size = 2, stride = 2)
        self.upconv1 = double_conv(in_c = 1024, out_c = 512) 
        self.tconv2 = nn.ConvTranspose2d(in_channels = 512, out_channels = 256, kernel_size = 2, stride = 2)
        self.upconv2 = double_conv(in_c = 512, out_c = 256) 
        self.tconv3 = nn.ConvTranspose2d(in_channels = 256, out_channels = 128, kernel_size = 2, stride = 2)
        self.upconv3 = double_conv(in_c = 256, out_c = 128) 
        self.tconv4 = nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = 2, stride = 2)
        self.upconv4 = double_conv(in_c = 128, out_c = 64)
        self.final = nn.Conv2d(in_channels = 64, out_channels = self.num_classes, kernel_size = 1)
        
    def forward(self, x):
        
        x1 = self.down_conv1(x) 
        x2 = self.maxpool(x1)
        x3 = self.down_conv2(x2) 
        x4 = self.maxpool(x3)
        x5 = self.down_conv3(x4) 
        x6 = self.maxpool(x5)
        x7 = self.down_conv4(x6) 
        x8 = self.maxpool(x7)
        x9 = self.down_conv5(x8)
        
        y = self.tconv1(x9)
        y1 = self.upconv1(torch.cat([crop(x7, y),y], dim = 1))
        y2 = self.tconv2(y1)
        y3 = self.upconv2(torch.cat([crop(x5,y2), y2], dim = 1))
        y4 = self.tconv3(y3)
        y5 = self.upconv3(torch.cat([crop(x3,y4), y4], dim = 1))
        y6 = self.tconv4(y5)
        y7 = self.upconv4(torch.cat([crop(x1,y6), y6], dim = 1))
        
        out = self.final(y7)
        return out


def test():
    
    ip = torch.randn((1,1,572,572))
    model = UNet(2)
    print(summary(model, (1, 572, 572), device = 'cpu'))
    print(model(ip).shape)
    

if __name__ == '__main__':
    
    test()