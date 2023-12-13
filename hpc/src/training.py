# Import necessary libraries
import torch
from torch import nn
import numpy as np
import torch.optim as optim
from tqdm import trange
from datetime import datetime
from data_process import getDataloaders
# Import from locally
from resnest import make_uresnet
from unet import make_unet
from u2net import make_u2net, multi_bce_loss_fusion


def train(model, train_dataloader, val_dataloader, optimizer, criterion=None, epo_num=10):
    """Training loop for three models: ResNet, UNet, U2Net

    Args:
        model (nn.Module): The model to train
        train_dataloader (dataloader): Dataloader holding the training data
        val_dataloader (dataloader): Dataloader holding the validation data
        optimizer (optim): Optimizer for the model (SDG or Adam)
        criterion (nn, optional): loss function for the model. Defaults to None, i.e. multi_bce_loss_fusion.
        epo_num (int, optional): Epochs to run for. Defaults to 10.

    Returns:
        model: the trained model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_name = model.name
    model = model.to(device)
    criterion = criterion.to(device) if criterion else None
    
    all_train_iter_loss = []
    all_val_iter_loss = []

    prev_time = datetime.now() # start timing
    for epo in trange(epo_num):
        
        train_loss = 0
        model.train()
        for _, (img, mask) in enumerate(train_dataloader):
            img, mask = img.to(device), mask.to(device) # img.shape [12, 3, 256, 256]
                                                        # mask.shape [12, 10, 256, 256]
             
            optimizer.zero_grad()
            if model_name == 'U2NET' or model_name == 'U2NET-small':    # Calculates loss differently
                d0, d1, d2, d3, d4, d5, d6 = model(img)
                _, loss = multi_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, mask)

            else:
                output = model(img)
                loss = criterion(output, mask)

            loss.backward()
            optimizer.step()
            
            # save loss of each iteration
            iter_loss = loss.item()
            all_train_iter_loss.append(iter_loss)
            train_loss += iter_loss
        
        # evaluate and save model each 10 epo
        if np.mod(epo, 10) == 0:
            val_loss = 0
            model.eval()
            with torch.no_grad():
                for _, (img, mask) in enumerate(val_dataloader):
                    img, mask = img.to(device), mask.to(device)
                    optimizer.zero_grad()
                    if model_name == 'U2NET' or model_name == 'U2NET-small':
                        d0, d1, d2, d3, d4, d5, d6 = model(img)
                        _, loss = multi_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, mask)
                    else:
                        output = model(img)
                        loss = criterion(output, mask)
                        
                    iter_loss = loss.item()
                    all_val_iter_loss.append(iter_loss)
                    val_loss += iter_loss
            
            # save model
            filename = f'{model_name}_{epo}_loss_trian_{round(train_loss/len(train_dataloader),5)}_val_{round(val_loss/len(val_dataloader),5)}.pt'
            torch.save(model, filename)
            print(f"\nSaving {filename}")

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        prev_time = cur_time

        print(f'\nepoch: {epo}/{epo_num}')
        print(f'\nepoch train loss = {train_loss/len(train_dataloader)}\nepoch\
              val loss = {val_loss/len(val_dataloader)}, {time_str}')

    return model


if __name__ == "__main__":
    tload, vload = getDataloaders('/zhome/4e/8/181483/deep-learning-project/data/new_dataset.npy', batch_size=32)
    model = make_uresnet()                                # <<< CHANGE MODEL HERE!! make_uresnest(enc_layers="50"/"101"/"200"),
                                                          # make_unet() or make_u2net()
    criterion = nn.BCELoss()                              # <<< CHANGE LOSS HERE!! nn.CrossEntropyLoss() or nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.7) # <<< CHANGE OPTIMIZER HERE!! optim.Adam() or optim.SGD()
    epochs = 100                                          # <<< CHANGE EPOCHS HERE!!
    
    model = train(model, tload, vload, optimizer, criterion=criterion, epo_num=epochs) 
