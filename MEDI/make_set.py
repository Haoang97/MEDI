import torch
import torch.nn as nn
import torch.utils.data as data
#from .utils import download_url, check_integrity
import torchvision.transforms as transforms
from PIL import Image

class make_set(data.Dataset):
    def __init__(self, data, label):
        self.data = data # array
        self.label = label # list
        
    def __getitem__(self, index):
        img, target = self.data[index], self.label[index]
        #img = img.type(torch.FloatTensor)
        #img = Image.fromarray(img)
        #img = self.transform(img)
        return img, target
    def __len__(self):
        return len(self.data)

        