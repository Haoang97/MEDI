import torch
import numpy as np
from pre_sample import cata
import random 
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_tensor
from PIL import Image

class CategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per, start):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        self.m_ind = []
        for i in range(start, max(label)+1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            for c in classes:
                l = self.m_ind[c]
                random.shuffle(l)
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch

class CategoriesSampler_v2():

    def __init__(self, data, label, n_batch, num_labeled_classes, n_cls, n_per, start, name):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per
        
        if name == 'cifar10':
            n_data = torch.FloatTensor(len(data),3,32,32)
            transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        elif name == 'cifar100':
            n_data = torch.FloatTensor(len(data),3,32,32)
            transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))])
        elif name == 'svhn':
            n_data = torch.FloatTensor(len(data),3,32,32)
            transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        elif name == 'omniglot':
            n_data = torch.FloatTensor(len(data),1,32,32)
            transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.086,), (0.235,))])
            
        for idx, i in enumerate(data):
            #print(type(i))
            if name != 'omniglot':
                i = Image.fromarray(i.transpose((1, 2, 0)))
            else:
                i = Image.fromarray(np.array(i).transpose((1,2,0)).squeeze(axis=2))
            i = transform(i)
            n_data[idx] = i

        self.img2label = {}
        for idx, img in enumerate(n_data):
            self.img2label[''.join(str(i) for i in img[0][10])] = idx

        self.data, self.labels = cata(n_data, [item-start for item in label], num_labeled_classes, name)

        self.all_ind =[]

        for j in range(3):
            #label = np.array(self.labels[i])
            label = self.labels[j]
            self.m_ind = []
            for i in range(0, torch.max(label)+1):
                ind = np.argwhere(label == i).reshape(-1)
                #ind = torch.from_numpy(ind)
                self.m_ind.append(ind)
            self.all_ind.append(self.m_ind)

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            if (i_batch+1)%3==1:
                idx=0
            elif (i_batch+1)%3==2:
                idx=1
            elif (i_batch+1)%3==0:
                idx=2
            batch = []
            classes = torch.randperm(len(self.all_ind[idx]))[:self.n_cls]
            for c in classes:
                l = self.all_ind[idx][c]
                #random.shuffle(l)
                #pos = torch.randperm(len(l))[:self.n_per]
                pos = np.random.choice(l, self.n_per, True)
                #batch.append(l[pos])
                batch.append(torch.tensor(pos))
            batch = torch.stack(batch).t().reshape(-1)
            new_batch = []
            for j in batch:
                new_idx = self.img2label[''.join(str(i) for i in self.data[idx][j][0][10])]
                new_batch.append(new_idx)
            yield new_batch