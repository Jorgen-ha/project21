import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
"""
With inspiration from https://www.youtube.com/watch?v=IHq1t7NxS8k&ab_channel=AladdinPersson
His implementation can be found here:  
https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/image_segmentation/semantic_segmentation_unet
"""

#Architecture of U_net
class DoubleConv(nn.Module):    # The double convolution block
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1,bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module): # The U-Net architecture
    def __init__(self, in_channels=3, out_channels=10, features = [64, 128, 256, 512]):
        """Function to initialize the U-Net model

        Args:
            in_channels (int, optional): input channels. Defaults to 3 (RGB).
            out_channels (int, optional): output channels. Defaults to 10.
            features (list, optional): list defining the number of features in each layer. Defaults to [64, 128, 256, 512].
        """
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.name = "UNET"

        # Down part of U-Net
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of U-Net
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))
        
        # Bottleneck layer
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)  #1024
        
        # Final layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(features[0], out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            # Check that we have the right shapes
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size= skip_connection.shape[2:])
            concat_skip = torch.cat((skip_connection, x), dim = 1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)
    
def make_unet():
    """Function creating the U-Net model"""
    return UNET(in_channels=3, out_channels=10)