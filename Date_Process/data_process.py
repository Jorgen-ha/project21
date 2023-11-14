# -*- coding: utf-8 -*-
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage import transform, img_as_ubyte
import glob
import torch
from torchvision import transforms
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset, DataLoader, random_split


#Convert a labeled image to a color image
def label_2_colour(label_image):
    l2c_dict = {
        # Label: (R, G, B) color
        0: (0, 0, 0),            # 0: Range Hood
        10: (250, 150, 10),      # 10: Orange Hood
        20: (20, 100, 20),       # 20: Dark Green Front Door
        30: (250, 250, 10),      # 30: Yellow Rear Door
        40: (10, 250, 250),      # 40: Cyan Frame
        50: (150, 10, 150),      # 50: Purple Rear Quarter Panel
        60: (10, 250, 10),       # 60: Light Green Trunk Lid
        70: (20, 20, 250),       # 70: Blue Fender
        80: (250, 10, 250),      # 80: Pink Bumper
        90: (0, 0, 0),           # 90: No Color (Rest of Car)
    }
    # Create an empty color image with the same dimensions as the label image
    color_image = np.zeros((label_image.shape[0], label_image.shape[1], 3), dtype=np.uint8)

    # Map labels to colors and fill the color image accordingly
    for label, color in l2c_dict.items():
            color_image[label_image == label] = color

    return color_image


#Add the segmentation result on the dealed image(size is 256x256)
def result_2_image(result_array):
    result_image = result_array[:,:,0:3]
    label_image = result_array[:,:,3]
    color_image = label_2_colour(label_image)
    mask = np.all(color_image != [0, 0, 0], axis=-1)
    result_image[mask] = color_image[mask]
    return result_image

#Resize an image to fit within a 256x256 canvas while preserving its aspect ratio.
def resize_2_256(image):
    width, height = image.shape[1], image.shape[0]
    aspect_ratio = width / height

    if aspect_ratio > 1:  # Width > Height
        new_width = 256
        new_height = int(256 / aspect_ratio)
    else:  # Height > Width
        new_height = 256
        new_width = int(256 * aspect_ratio)
    resized_image = transform.resize(image, (new_height, new_width))
    canvas = np.zeros((256, 256, 3), dtype=np.uint8)
    canvas[0:new_height, 0:new_width] = (resized_image * 255).astype(np.uint8)

    return canvas

#Resize a segmentation result image to match the original image's size and overlay it.
def resize_result_to_original(original_image, result_array):
    color_image = label_2_colour(result_array[:, :, -1])
    result_image = original_image.copy()

    # Determine the new dimensions while preserving aspect ratio
    width, height = result_image.shape[1], result_image.shape[0]
    aspect_ratio = width / height

    if aspect_ratio > 1:  # Width > Height
        new_width = 256
        new_height = int(256 / aspect_ratio)
    else:  # Height > Width
        new_height = 256
        new_width = int(256 * aspect_ratio)

    # Crop the color image to fit the new dimensions
    cropped_image = color_image[0:new_height, 0:new_width]

    # Resize the cropped image to match the original image's size
    segmentation_result = transform.resize(cropped_image, original_image.shape[:2], mode='constant', anti_aliasing=False)
    segmentation_result = img_as_ubyte(segmentation_result)

    # Generate a mask to identify non-black (non-transparent) pixels in the segmentation result
    mask = np.all(segmentation_result != [0, 0, 0], axis=-1)

    # Overlay the resized segmentation result on the original image
    result_image[mask] = segmentation_result[mask]

    return result_image


def onehot(data):
    categories = [[0, 10, 20, 30, 40, 50, 60, 70, 80, 90]]
    encoder = OneHotEncoder(categories=categories, sparse_output=False)
    data_flat = data.ravel()
    onehot_encoded = encoder.fit_transform(data_flat.reshape(-1, 1))
    onehot_encoded = onehot_encoded.reshape(256, 256, -1)
    
    return onehot_encoded


class Deloitte_Dataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.transform  = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(
                                            mean=[0.485, 0.456, 0.406], 
                                            std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        image = sample[:,:,:3]  #rgd image
        label = sample[:,:,3]  #label image

        # change date type form numpy to tensor
        if self.transform is not None:
            image = self.transform(image)

        label = onehot(label) # (n,256,256,10)
        label = label.transpose(2,0,1)#(n,10, 256,256)
        label = torch.FloatTensor(label)

        return image, label



def make_datasets(array_path):
    array_names = glob.glob(array_path + '*.npy')
    array_names.sort()
    print(f"Number of arrays: {len(array_names)}")
    dataset = []
    
    for array in array_names:
        dataset.append(np.load(array))
    
    dataset = Deloitte_Dataset(np.array(dataset))
    torch.save(dataset, 'dataset{}.pth'.format(len(array_names)))
    
    return dataset
    
