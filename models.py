import torch
import torch.nn as nn
import numpy as np

class convconv(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(convconv, self).__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias = False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias = False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

  def forward(self, x):
    return self.conv(x)

class smallUNet(nn.Module):
  def __init__(self, in_channels = 3, out_channels = 3, features = [12, 24, 48]):
    super(smallUNet, self).__init__()

    self.enter = convconv(in_channels, features[0])
    self.down1 = convconv(features[0], features[1])    
    self.pool = nn.MaxPool2d(2, 2)

    self.bottle = convconv(features[1], features[2])
    
    self.unconv1 = nn.Sequential(
      nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
    )
    
    self.unconv2 = nn.Sequential(
      nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
    )

    self.up1 = convconv(features[1]+features[1], features[1])
    self.up2 = convconv(features[0]+features[0], features[0])

    self.final = nn.Conv2d(features[0], out_channels, 1)

  def forward(self, x):
    out = self.enter(x)
    # down
    cat1 = out
    out = self.pool(out)
    out = self.down1(out)
    cat2 = out
    out = self.pool(out)
    # bottleneck
    out = self.bottle(out)
    # up
    out = self.unconv1(out)
    out = self.up1(torch.cat((cat2, out), dim = 1))
    out = self.unconv2(out)
    out = self.up2(torch.cat((cat1, out), dim = 1))
    out = self.final(out)
    # print("final shape", out.shape)
    return out

class LUNet(nn.Module):
  def __init__(self, in_channels = 3, out_channels = 3, features = [32, 64, 128, 256, 512]):
    super(LUNet, self).__init__()

    self.enter = convconv(in_channels, features[0])
    self.down1 = convconv(features[0], features[1])
    self.down2 = convconv(features[1], features[2])
    self.down3 = convconv(features[2], features[3])
    self.pool = nn.MaxPool2d(2, 2)

    self.bottle = convconv(features[3], features[4])
    
    self.unconv1 = nn.Sequential(
      nn.ConvTranspose2d(features[4], features[3], kernel_size=2, stride=2)
    )
    
    self.unconv2 = nn.Sequential(
      nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
    )
    
    self.unconv3 = nn.Sequential(
      nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
    )
    
    self.unconv4 = nn.Sequential(
      nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
    )

    self.up1 = convconv(features[3]+features[3], features[3])
    self.up2 = convconv(features[2]+features[2], features[2])
    self.up3 = convconv(features[1]+features[1], features[1])
    self.up4 = convconv(features[0]+features[0], features[0])

    self.final = nn.Conv2d(features[0], out_channels, 1)

  def forward(self, x):
    out = self.enter(x)
    # down
    cat1 = out
    out = self.pool(out)
    out = self.down1(out)
    cat2 = out
    out = self.pool(out)
    out = self.down2(out)
    cat3 = out
    out = self.pool(out)
    out = self.down3(out)
    cat4 = out
    out = self.pool(out)
    # bottleneck
    out = self.bottle(out)
    # up
    out = self.unconv1(out)
    out = self.up1(torch.cat((cat4, out), dim = 1))
    out = self.unconv2(out)
    out = self.up2(torch.cat((cat3, out), dim = 1))
    out = self.unconv3(out)
    out = self.up3(torch.cat((cat2, out), dim = 1))
    out = self.unconv4(out)
    out = self.up4(torch.cat((cat1, out), dim = 1))
    out = self.final(out)
    # print("final shape", out.shape)
    return out

class mUnet(nn.Module):
  def __init__(self, in_channels = 3, out_channels = 3, features = [16, 32, 64, 128]):
    super(mUnet, self).__init__()

    self.enter = convconv(in_channels, features[0])
    self.down1 = convconv(features[0], features[1])
    self.down2 = convconv(features[1], features[2])
    self.pool = nn.MaxPool2d(2, 2)

    self.bottle = convconv(features[2], features[3])
    
    self.unconv1 = nn.Sequential(
      nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
    )
    
    self.unconv2 = nn.Sequential(
      nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
    )
    
    self.unconv3 = nn.Sequential(
      nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
    )

    self.up1 = convconv(features[2]+features[2], features[2])
    self.up2 = convconv(features[1]+features[1], features[1])
    self.up3 = convconv(features[0]+features[0], features[0])

    self.final = nn.Conv2d(features[0], out_channels, 1)

  def forward(self, x):
    out = self.enter(x)
    # down
    cat1 = out
    out = self.pool(out)
    out = self.down1(out)
    cat2 = out
    out = self.pool(out)
    out = self.down2(out)
    cat3 = out
    out = self.pool(out)
    # bottleneck
    out = self.bottle(out)
    # up
    out = self.unconv1(out)
    out = self.up1(torch.cat((cat3, out), dim = 1))
    out = self.unconv2(out)
    out = self.up2(torch.cat((cat2, out), dim = 1))
    out = self.unconv3(out)
    out = self.up3(torch.cat((cat1, out), dim = 1))
    out = self.final(out)
    # print("final shape", out.shape)
    return out

class SimpleNet(nn.Module):
  def __init__(self, in_channels = 3, out_channels = 3, in_size = (32, 32), out_size = (32, 32)):
    super(SimpleNet, self).__init__()

    self.layer1 = nn.Linear(in_size[0]*in_size[1]*in_channels, out_size[0]*out_size[1]*out_channels);

  def forward(self, x):
    if(len(x.shape) > 2):
      x = x.reshape((x.shape[0], 3*32*32)) # hardcoded
    # print("x shape", x.shape)
    out = self.layer1(x);
    return out


class LinearNet(nn.Module):
  def __init__(self, in_channels = 3, out_channels = 3, in_size = (32, 32), out_size = (32, 32), middle_size = 256):
    super(LinearNet, self).__init__()

    self.layer1 = nn.Linear(in_size[0]*in_size[1]*in_channels, 256)
    self.r1 = nn.ReLU(inplace=True)
    self.middle = nn.Sequential(
        nn.Linear(256, 96),
        nn.ReLU(inplace=True)
        )
    self.middle2 = nn.Sequential(
        nn.Linear(96, 256),
        nn.ReLU(inplace=True))
    self.middle3 = nn.Linear(256, out_size[0]*out_size[1]*out_channels)

  def forward(self, x):
    if(len(x.shape) > 2):
      x = x.reshape((x.shape[0], 3*32*32)) # hardcoded
    # print("x shape", x.shape)
    out = self.layer1(x)
    out = self.middle(out)
    out = self.middle2(out)
    out = self.middle3(out)

    return out

class smallConvNet(nn.Module):
  def __init__(self, in_channels = 3, out_channels = 3, features = 12):
    super(smallConvNet, self).__init__()
    self.c1 = nn.Sequential(
      nn.Conv2d(in_channels, features, 3, 1, 1, bias = False),
      nn.BatchNorm2d(features),
      nn.ReLU(inplace=True),
      # nn.Conv2d(features, out_channels, 3, 1, 1, bias = False)
    )
    self.pool = nn.MaxPool2d(2, 2)
    self.c2 = nn.ConvTranspose2d(
              features, out_channels, kernel_size=2, stride=2
          )
    
  def forward(self, x):
    out = self.c1(x)
    # print('out.shape', out.shape)
    out = self.pool(out)
    # print('out.shape', out.shape)
    out = self.c2(out)
    # print('out.shape', out.shape)
    return out