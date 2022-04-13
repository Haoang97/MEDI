from model.resnet import *
from model.convnet import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from make_set import make_set
import sys
sys.path.append('..')
from utils import calc_similiar_penalty, AverageMeter, init_cluster
import numpy as np

def beta(epoch):
    return 0.9**(epoch//5)

def cata(data, label, num_labeled_classes, name):
    device = torch.device('cuda')
    criterion = nn.CrossEntropyLoss()

    dataset = make_set(data, label)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=512, shuffle=True)
    
    if name in ['cifar10','cifar100','svhn']:
        Encoder = ResNet(BasicBlock, [2, 2, 2, 2]).to(device)
        Head_1 = Classifier(num_classes=num_labeled_classes, use_BN=True).to(device)
        Head_2 = Classifier(num_classes=num_labeled_classes, use_BN=True).to(device)
        Head_3 = Classifier(num_classes=num_labeled_classes, use_BN=True).to(device)
    elif name == 'omniglot':
        Encoder = Omninet(1,64, 64).to(device)
        Head_1 = Omni_Classifier(num_labeled_classes).to(device)
        Head_2 = Omni_Classifier(num_labeled_classes).to(device)
        Head_3 = Omni_Classifier(num_labeled_classes).to(device)
    else:
        print('Dataset error.')

    opt_all = optim.Adam(list(Encoder.parameters())+list(Head_1.parameters())+list(Head_2.parameters())+
                            list(Head_3.parameters()), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-5)
    opt_head = optim.SGD(list(Head_1.parameters())+list(Head_2.parameters())+
                            list(Head_3.parameters()), lr=1e-3, momentum=0.9, weight_decay=1e-4)
    #opt_Head1 = optim.SGD(Head_1.parameters(), lr=1e-3, momentum=0.9)
    #opt_Head2 = optim.SGD(Head_2.parameters(), lr=1e-3, momentum=0.9)
    #opt_Head3 = optim.SGD(Head_3.parameters(), lr=1e-3, momentum=0.9)
    
    #opt_all_scheduler = torch.optim.lr_scheduler.StepLR(opt_all, step_size=50, gamma=0.1)
    opt_head_scheduler = torch.optim.lr_scheduler.StepLR(opt_head, step_size=150, gamma=0.1)
    
    for epoch in range(200):
        loss_record = AverageMeter()
        for step, (images, labels) in enumerate(dataloader):
            images = images.type(torch.FloatTensor)
            images, labels = images.to(device), labels.to(device)

            features = Encoder(images)
            output1 = Head_1(features)
            output2 = Head_2(features)
            output3 = Head_3(features)

            loss_similiar_12 = calc_similiar_penalty(Head_1, Head_2)
            loss_similiar_13 = calc_similiar_penalty(Head_1, Head_3)
            loss_similiar_23 = calc_similiar_penalty(Head_2, Head_3)

            loss_1 = criterion(output1, labels)
            loss_2 = criterion(output2, labels)
            loss_3 = criterion(output3, labels)
            
            if epoch <= 100:
                loss = loss_1 + loss_2 + loss_3
                opt_all.zero_grad()
                loss.backward() 
                opt_all.step()
                
            else:
                loss = 0.2*(loss_1 + loss_2 + loss_3) + loss_similiar_12 + loss_similiar_13 + loss_similiar_23
                opt_head.zero_grad()
                loss.backward()
                opt_head.step()

            loss_record.update(loss.item(), images.size(0))

        #opt_all_scheduler.step()
        opt_head_scheduler.step()
            
        print('Train Epoch: {} Avg Loss: {:.4f}'.format(epoch, loss_record.avg))

    cluster_data = [[] for i in range(3)]
    cluster_label = [[] for i in range(3)]
    cluster_data, cluster_label = init_cluster(dataset, cluster_data, cluster_label, num_labeled_classes)
    dataloader_eval = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    
    Encoder.eval()
    Head_1.eval()
    Head_2.eval()
    Head_3.eval()

    for step, (image, label) in enumerate(dataloader_eval):
        image = image.type(torch.FloatTensor)
        image, label = image.to(device), label.to(device)

        features = Encoder(image)
        output1 = Head_1(features)
        output2 = Head_2(features)
        output3 = Head_3(features)

        #output1, output2, output3 = output1.squeeze(0), output2.squeeze(0), output3.squeeze(0)
        prob1, prob2, prob3 = output1.squeeze(0)[label], output2.squeeze(0)[label], output3.squeeze(0)[label]
       
        
        #logits1, _ = torch.max(output1, 0)
        #logits2, _ = torch.max(output2, 0)
        #logits3, _ = torch.max(output3, 0)
        
        #_, assign_class = torch.max(torch.stack((logits1,logits2,logits3), 0),0)
        _, assign_class = torch.max(torch.stack((prob1,prob2,prob3), 0),0)
        
        cluster_data[assign_class.item()].append(image.squeeze(axis=0).cpu())# the type of image is "tensor"
        cluster_label[assign_class.item()].append(label.cpu().item())

    for k in range(3):
        cluster_data[k] = torch.tensor([item.numpy() for item in cluster_data[k]])
        cluster_label[k] = torch.LongTensor([item for item in cluster_label[k]])

    #cluster_data[0], cluster_data[1], cluster_data[2] = np.array(cluster_data[0]), np.array(cluster_data[1]), np.array(cluster_data[2])
    #cluster_label[0], cluster_label[1], cluster_label[2] = np.array(cluster_label[0]), np.array(cluster_label[1]), np.array(cluster_label[2])
    #print(cluster_data.shape)
    #print(cluster_label)
    print('num_class1:{}, num_class2:{}, num_class3:{}'.format(len(cluster_label[0]), len(cluster_label[1]), len(cluster_label[2])))

    return cluster_data, cluster_label # the type of cluster_data is [tensor(tensor0,tensor1,...),tensor(tensor0,tensor1,...),tensor(tensor0,tensor1,...)]















    

