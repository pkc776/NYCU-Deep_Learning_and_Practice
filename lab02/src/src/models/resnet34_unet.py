import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Basic Residual Block for ResNet34"""
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet34Encoder(nn.Module):
    """ResNet34 Encoder for U-Net"""
    def __init__(self, in_channels=3):
        super(ResNet34Encoder, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet34 layers: [3, 4, 6, 3]
        self.layer1 = self._make_layer(64, 64, 3, stride=1)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Initial convolution
        x1 = F.relu(self.bn1(self.conv1(x)))  # 64 channels
        x1_pool = self.maxpool(x1)  # 64 channels
        
        # ResNet layers
        x2 = self.layer1(x1_pool)  # 64 channels
        x3 = self.layer2(x2)       # 128 channels
        x4 = self.layer3(x3)       # 256 channels
        x5 = self.layer4(x4)       # 512 channels
        
        return [x1, x2, x3, x4, x5]


class UNetDecoder(nn.Module):
    """U-Net Decoder with skip connections"""
    def __init__(self, num_classes=3):
        super(UNetDecoder, self).__init__()
        
        # Upsampling layers
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv1 = self._double_conv(256 + 256, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = self._double_conv(128 + 128, 128)
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = self._double_conv(64 + 64, 64)
        
        self.up4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv4 = self._double_conv(64 + 64, 64)
        
        self.final_up = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.final_conv = self._double_conv(64, 64)

        # Final output layer
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def _double_conv(self, in_channels, out_channels):
        """Double convolution layer"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, features):
        x1, x2, x3, x4, x5 = features
        
        # Upsampling with skip connections
        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.conv1(x)
        
        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv2(x)
        
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv3(x)
        
        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv4(x)
        
        x = self.final_up(x)
        x = self.final_conv(x)
        
        # Final output
        x = self.out_conv(x)
        return x


class ResNet34UNet(nn.Module):
    """ResNet34 + U-Net hybrid architecture"""
    def __init__(self, in_channels=3, num_classes=3):
        super(ResNet34UNet, self).__init__()
        
        self.encoder = ResNet34Encoder(in_channels)
        self.decoder = UNetDecoder(num_classes)
        
    def forward(self, x):
        features = self.encoder(x)
        output = self.decoder(features)
        return output
