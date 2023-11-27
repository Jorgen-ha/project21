# Script to train ResNeSt model
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import OneHotEncoder
import torch.optim as optim
from tqdm import trange
from datetime import datetime


def onehot(data):
    categories = [[0, 10, 20, 30, 40, 50, 60, 70, 80, 90]]
    encoder = OneHotEncoder(categories=categories, sparse_output=False)
    data_flat = data.ravel()
    onehot_encoded = encoder.fit_transform(data_flat.reshape(-1, 1))
    onehot_encoded = onehot_encoded.reshape(256, 256, -1)
    
    return onehot_encoded

class MyDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        image = sample[:,:,:3]  #rgb image
        label = sample[:,:,3]  #label image

        # change date type form numpy to tensor
        if self.transform is not None:
            image = self.transform(image)

        label = onehot(label) # (n,256,256,10)
        label = label.transpose(2,0,1)#(n,10, 256,256)
        label = torch.FloatTensor(label)

        return image, label

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
        # x = self.bn2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        
        if self.downsample is not None:
            residual = self.downsample(residual)

        x += residual
        x = self.relu(x)
        
        return x
class ResNeSt(nn.Module):
    def __init__(self, name, block, layers, img_c, n_classes):
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
        self.layer1 = self._make_layer(block, layers[0], out_c=64, radix=2, cardinality=1, stride=1)
        self.layer2 = self._make_layer(block, layers[1], out_c=128, radix=2, cardinality=1, stride=2)
        self.layer3 = self._make_layer(block, layers[2], out_c=256, radix=2, cardinality=1, stride=2)
        self.layer4 = self._make_layer(block, layers[3], out_c=512, radix=2, cardinality=1, stride=2)
        
        self.dropout = nn.Dropout(p=0.2)    # as per paper
        self.final_conv = nn.Conv2d(512*block.expansion, n_classes, kernel_size=1)
        self.upsample = nn.Upsample((256, 256), mode='bilinear', align_corners=True)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels # He initialisation
                m.weight.data.normal_(0, np.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)  # Initialise BN as per paper
                m.bias.data.zero_()
        
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

        x = self.conv1(x)   # initial layer
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)    

        x = self.layer1(x) # ResNeSt layers
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.dropout(x)

        x = self.final_conv(x)
        x = self.upsample(x)
        
        return x
    
class DiceLoss(nn.Module):
    def forward(self, input, target):
        smooth = 1.
        iflat = input.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()
        loss = 1 - ((2. * intersection + smooth) /
                    (iflat.sum() + tflat.sum() + smooth))
        return loss
    
    
def resnest50(in_channels=3, out_channels=10):
    return ResNeSt("50", Block, [3, 4, 6, 3], img_c=in_channels, n_classes=out_channels)

def resnest101(in_channels=3, out_channels=10):
    return ResNeSt("101", Block, [3, 4, 23, 3], img_c=in_channels, n_classes=out_channels)

def resnest200(in_channels=3, out_channels=10):
    return ResNeSt("200", Block, [3, 24, 36, 3], img_c=in_channels, n_classes=out_channels)
    
    
def train(train_dataloader, val_dataloader, epo_num=10, model_name='resNeSt50'):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = resnest50(in_channels = 3, out_channels = 10)#input is rgb output is 10 classes
    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device) #loss
    optimizer = optim.Adam(model.parameters(), lr=1e-5) #optimizer
    # optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.7) #optimizer
    
    
    all_train_iter_loss = []
    all_val_iter_loss = []

    # start timing
    prev_time = datetime.now()
    for epo in trange(epo_num):

        train_loss = 0
        model.train()
        for _, (car, car_msk) in enumerate(train_dataloader):
            car = car.to(device)            # car.shape is torch.Size([12, 3, 256, 256])
            car_msk = car_msk.to(device)    # car_msk.shape is torch.Size([12, 10, 256, 256])    
                                                    
            optimizer.zero_grad()
            output = model(car)
            loss = criterion(output, car_msk)
            loss.backward()
            iter_loss = loss.item()
            all_train_iter_loss.append(iter_loss)
            train_loss += iter_loss
            optimizer.step()
        
        # evaluate each 5 epo
        if np.mod(epo, 5) == 0:
            val_loss = 0
            model.eval()

            with torch.no_grad():
                for _, (car, car_msk) in enumerate(val_dataloader):
                    car = car.to(device)
                    car_msk = car_msk.to(device)
                    optimizer.zero_grad()
                    output = model(car) 
                    loss = criterion(output, car_msk)
                    iter_loss = loss.item()
                    all_val_iter_loss.append(iter_loss)
                    val_loss += iter_loss
            
            # save model each 5 epoch
            filename = f'{model_name}_{epo}_loss_trian_{round(train_loss/len(train_dataloader), 5)}\
                         _val_{round(val_loss/len(val_dataloader), 5)}.pt'
            torch.save(model, filename)
            print(f"\nSaving {filename}")

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        prev_time = cur_time

        print(f'\nepoch: {epo}/{epo_num}')
        print(f'\nepoch train loss = {train_loss/len(train_dataloader)}\nepoch val loss = \
                {val_loss/len(val_dataloader)}, {time_str}')

    return model


# if __name__ == "__main__":
#     dataset = np.load('/zhome/4e/8/181483/deep-learning-project/dataset.npy') # Load dataset
#     transform = transforms.Compose([transforms.ToTensor(),
#                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], 
#                                                           std=[0.229, 0.224, 0.225])]) #normalization
#     train_set = MyDataset(dataset, transform)
#     train_size = int(0.95 * len(train_set))   # 95% for train
#     val_size = len(train_set) - train_size    # 5% for validation
#     train_dataset, val_dataset = random_split(train_set, [train_size, val_size])

#     tload = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
#     vload = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=2)
    
#     model = train(tload, vload, epo_num=6, model_name='resNeSt50') 
    