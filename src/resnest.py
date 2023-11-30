import torch
import torch.nn as nn

class rSoftmax(nn.Module):
    """rSoftmax module, as per the paper"""
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality
    
    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1,2)
            x = x.softmax(dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        
        return x

# Blocks to have ResNet become ResNeSt
class SplitAttention(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, radix=2, cardinality=1, bias=True, stride=1, padding=0):
        """Initialises a SplitAttention block, as part of the ResNeSt block

        Args:
            in_c (int): input channels
            radix (int): number of splits within a cardinal group, denoted r
            cardinality (int): number of feature map groups, denoted k
        """
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality
        self.channels = out_c
        self.conv = nn.Conv2d(in_c, out_c*radix, kernel_size=kernel_size, groups=cardinality*radix, 
                              bias=bias, padding=padding, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_c*radix)
        self.bn2 = nn.BatchNorm2d(out_c*radix)
        self.fc1 = nn.Conv2d(out_c, out_c*radix, kernel_size=1, groups=cardinality)
        self.fc2 = nn.Conv2d(out_c*radix, out_c*radix, kernel_size=1, groups=cardinality)
        self.relu = nn.ReLU(inplace=True)
        self.rsoftmax = rSoftmax(radix, cardinality)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            splited = torch.split(x, int(rchannel//self.radix), dim=1)
            gap = sum(splited)
        else:
            gap = x
        gap = nn.functional.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)
        gap = self.bn2(gap)
        gap = self.relu(gap)
        
        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)
        
        if self.radix > 1:
            attens = torch.split(atten, int(rchannel//self.radix), dim=1)
            out = sum([att*split for (att, split) in zip(attens, splited)])
        else:
            out = atten * x
        
        return out.contiguous()
    
class Block(nn.Module):     # ResNeSt block (bottleneck)
    expansion = 4
    def __init__(self, in_c, out_c, radix=2, cardinality=1, downsample=None, stride=1):
        """Initialises the ResNeSt block

        Args:
            in_c (int): input channels
            out_c (int): output channels
            radix (int): number of splits within a cardinal group, denoted R
            cardinality (int): number of cardinal groups, denoted K
            downsample (_type_, optional): _description_. Defaults to None.
            stride (int, optional): stride. Defaults to 1.
        """
        super().__init__()
        group_width = int(out_c * cardinality)
        self.conv1 = nn.Conv2d(in_c, group_width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(group_width)
        self.radix = radix
        if radix >= 1:  
            self.conv2 = SplitAttention(group_width, group_width, kernel_size=3, radix=radix, 
                                        cardinality=cardinality, bias=False, padding=1, stride=stride)
        else:
            self.conv2 = nn.Conv2d(group_width, group_width, kernel_size=3, 
                                   cardinality=cardinality, bias=False, padding=1, stride=stride)
            self.bn2 = nn.BatchNorm2d(group_width)

        self.conv3 = nn.Conv2d(group_width, out_c*self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_c*self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        if self.radix == 0:
            x = self.bn2(x)
            x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        
        if self.downsample is not None:
            residual = self.downsample(residual)

        x += residual
        x = self.relu(x)
        
        return x
        

class Block(nn.Module):     # ResNeSt block (bottleneck)
    expansion = 4
    def __init__(self, in_c, out_c, radix=2, cardinality=1, downsample=None, stride=1):
        """Initialises the ResNeSt block

        Args:
            in_c (int): input channels
            out_c (int): output channels
            radix (int): number of splits within a cardinal group, denoted R
            cardinality (int): number of cardinal groups, denoted K
            downsample (_type_, optional): _description_. Defaults to None.
            stride (int, optional): stride. Defaults to 1.
        """
        super().__init__()
        group_width = int(out_c * cardinality)
        self.conv1 = nn.Conv2d(in_c, group_width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(group_width)
        self.radix = radix
        if radix >= 1:  
            self.conv2 = SplitAttention(group_width, group_width, kernel_size=3, radix=radix, 
                                        cardinality=cardinality, bias=False, padding=1, stride=stride)
        else:
            self.conv2 = nn.Conv2d(group_width, group_width, kernel_size=3, 
                                   cardinality=cardinality, bias=False, padding=1, stride=stride)
            self.bn2 = nn.BatchNorm2d(group_width)

        self.conv3 = nn.Conv2d(group_width, out_c*self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_c*self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        if self.radix == 0:
            x = self.bn2(x)
            x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        
        if self.downsample is not None:
            residual = self.downsample(residual)

        x += residual
        x = self.relu(x)
        
        return x


class ResNeSt(nn.Module): # Encoder
    def __init__(self, name, block, layers, img_c):
        """Initialises the ResNeSt model

        Args:
            block (nn.Module): The ResNeSt block
            layers (list): holding the number of blocks in each layer
            img_c (int): input channels
            n_classes (int): output classes
        """
        self.cardinality = 1
        self.group_width = 64
        self.name = f"ResNeSt{name}"

        super().__init__()
        # Initial layer, same as for ResNet this is NOT a ResNeSt layer
        self.conv1 = nn.Conv2d(img_c, self.group_width, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.group_width)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNeSt layers
        self.layer1 = self._make_layer(block, layers[0], out_c=32, radix=2, cardinality=1, stride=1)
        self.layer2 = self._make_layer(block, layers[1], out_c=64, radix=2, cardinality=1, stride=2)
        self.layer3 = self._make_layer(block, layers[2], out_c=128, radix=2, cardinality=1, stride=2)
        self.layer4 = self._make_layer(block, layers[3], out_c=256, radix=2, cardinality=1, stride=2)


    def _make_layer(self, block, n_blocks, out_c, radix, cardinality, stride):
        """Internal function to create the ResNeSt layers

        Args:
            block (Block): convolutional block as per the ResNet architecture
            n_res_blocks (int): number of residual blocks, number of times blocks are used
            out_c (int): number of channels when done with this layer
            radix (int): number of splits within a cardinal group, denoted R
            cardinality (int): number of cardinal groups, denoted K
            stride (int): 1 or 2 depending on the layer
        """
        downsample = None

        if stride != 1 or self.group_width != out_c * block.expansion:
            down_layers = []
            down_layers.append(nn.Conv2d(self.group_width, out_c * block.expansion, kernel_size=1,
                                         stride=stride, bias=False))
            down_layers.append(nn.BatchNorm2d(out_c * block.expansion))
            downsample = nn.Sequential(*down_layers)

        layers = []
        layers.append(block(self.group_width, out_c, radix, cardinality, downsample, stride))
        self.group_width = out_c * block.expansion
        for _ in range(1, n_blocks):
            layers.append(block(self.group_width, out_c, radix, cardinality))

        return nn.Sequential(*layers)

    def forward(self, x):
        skips = []
        x = self.conv1(x)   # initial layer
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # ResNeSt layers
        x = self.layer1(x); skips.append(x);
        x = self.layer2(x); skips.append(x);
        x = self.layer3(x); skips.append(x);
        x = self.layer4(x); skips.append(x);

        return skips


def resnest50(in_channels=3):
  return ResNeSt('50', Block, [3, 4, 6, 3], img_c=in_channels)

def resnest101(in_channels=3):
  return ResNeSt('101', Block, [3, 4, 23, 3], img_c=in_channels)

def resnest200(in_channels=3):
  return ResNeSt('200', Block, [3, 24, 36, 3], img_c=in_channels)

def make_resnest(layers="50"):
    if layers == "50":
        return resnest50()
    elif layers == "101":
        return resnest101()
    elif layers == "200":
        return resnest200()
    else:
        raise Exception("Invalid layers for ResNeSt")


#Architect of U_net - decoder
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class URESnet(nn.Module):
    def __init__(self, n_classes=10, encoder_layers="50"):
        super().__init__()
        self.name = "URESnet"
        self.encoder = make_resnest(encoder_layers)
        self.ups = nn.ModuleList()
        self.extracters = nn.ModuleList()

        self.bottleneck = DoubleConv(1024, 2048)
        # Four skip layers
        for feature in [1024, 512, 256, 128]:
            self.ups.append(nn.ConvTranspose2d(feature, feature//2, kernel_size=2, stride=2))
            self.extracters.append(DoubleConv(feature*2, feature))


        # Final layer
        self.final = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2), #64 -> 64 * 256
            nn.Conv2d(64, n_classes, kernel_size=1),#64
            nn.Sigmoid()
            )

    def forward(self, x):
        skips = self.encoder(x)

        x = skips[-1]       #1024 8 8
        skips = skips[::-1] #reverse the list
        x = self.ups[0](x)  #512 16 16
        for i in range(1,len(self.ups)):
            skip = skips[i]                  #512 16 16 #256 32 32 #128 64 64
            x = torch.cat((skip, x), dim = 1)# 1024 16 16 # 512 32 32  #256 64 64
            x = self.extracters[i](x)        # 512 16 16 #256 32 32 #128 64 64
            x = self.ups[i](x)               #256 32 32 #128 64 64 #64 128 128


        x = self.final(x)  # goes from [2, 128, 128, 128] --> [2, 10, 256, 256]

        return x

def make_uresnet(layers="50"):
    return URESnet(encoder_layers=layers)
    