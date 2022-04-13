import argparse
import os.path as osp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model.resnet import ResNet, BasicBlock, Projector
from model.convnet import Omninet
from data.svhn import My_SVHN
from data.cifar10 import My_CIFAR10
from data.cifar100 import My_CIFAR100
from data.omniglot import My_Omniglot
from samplers import CategoriesSampler, CategoriesSampler_v2
from utils import pprint, set_gpu, ensure_path, Averager, Timer, count_acc, euclidean_metric, cluster_acc, seed_torch
from torch.autograd import Variable
import numpy as np
from make_set import make_set

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=200)
    parser.add_argument('--save-epoch', type=int, default=20)
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--train-way', type=int, default=20)
    parser.add_argument('--test-way', type=int, default=20)
    parser.add_argument('--num_labeled_classes', type=int, default=80) #O: 964, S: 5, C10:5, C100: 80
    parser.add_argument('--num_unlabeled_classes', type=int, default=20) #O: 659, S: 5, C10: 5, C100: 20
    parser.add_argument('--save-path', default='./save/proto-1-c100')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--dataset', type=str, default='cifar100')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', default=0, type=int)
    args = parser.parse_args()
    pprint(vars(args))

    set_gpu(args.gpu)
    ensure_path(args.save_path)

    seed_torch(args.seed)
    
    
    if args.dataset == 'omniglot':
        trainset = My_Omniglot('train')
        train_data, train_label = [], []
        for i in trainset:
            train_data.append(i[0])
            train_label.append(int(i[1]))
        
        train_sampler = CategoriesSampler_v2(train_data, train_label, 100, args.num_labeled_classes,
                                        args.train_way, args.shot + args.query, 0, 'omniglot')
        train_sampler = CategoriesSampler(train_label,100,args.test_way,args.shot + args.query, 0)
        train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                                num_workers=8, pin_memory=True)
        valset = My_Omniglot('test')
        val_label =[]
        for i in valset:
            val_label.append(int(i[1]))
        val_sampler = CategoriesSampler(val_label, 400,
                                        args.test_way, args.shot + args.query, 0)
        val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                                num_workers=8, pin_memory=True)
        test_loader = DataLoader(dataset=valset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)
        model = Omninet(1, 64, 60).cuda()
    elif args.dataset == 'cifar10':
        trainset = My_CIFAR10('train')
        encoder = ResNet(BasicBlock, [2, 2, 2, 2])
        model = Projector(encoder, 512, 64)
        en_state = torch.load('./pretrain/Encoder/CIFAR10_E_checkpoint.tar',map_location='cpu')
        pr_state = torch.load('./pretrain/Projector/CIFAR10_P_checkpoint.tar',map_location='cpu')
        model.encoder.load_state_dict(en_state)
        model.projector.load_state_dict(pr_state)
        model = model.cuda()
        train_sampler = CategoriesSampler_v2(trainset.data, trainset.targets, 100, 5,
                                        args.train_way, args.shot + args.query, 0, args.dataset)
        train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                                num_workers=8, pin_memory=True)
        valset = My_CIFAR10('test')
        val_sampler = CategoriesSampler(valset.targets, 400,
                                        args.test_way, args.shot + args.query, 5)  
        val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                                num_workers=8, pin_memory=True) 
        test_loader = DataLoader(dataset=valset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)
        
    elif args.dataset == 'cifar100':
        trainset = My_CIFAR100('train')
        encoder = ResNet(BasicBlock, [2, 2, 2, 2])
        model = Projector(encoder, 512, 64)
        en_state = torch.load('./pretrain/Encoder/CIFAR100_E_checkpoint.tar',map_location='cpu')
        pr_state = torch.load('./pretrain/Projector/CIFAR100_P_checkpoint.tar',map_location='cpu')
        model.encoder.load_state_dict(en_state)
        model.projector.load_state_dict(pr_state)
        model = model.cuda()
        train_sampler = CategoriesSampler_v2(trainset.data, trainset.label, 100, args.num_labeled_classes,
                                        args.train_way, args.shot + args.query, 0, 'omniglot')
        train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                                num_workers=8, pin_memory=True)
        valset = My_CIFAR100('test')
        val_sampler = CategoriesSampler(valset.targets, 400,
                                        args.test_way, args.shot + args.query, 80)  
        val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                                num_workers=8, pin_memory=True)
        test_loader = DataLoader(dataset=valset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)
        #model = Convnet(3,64,60).cuda()
    elif args.dataset == 'svhn':
        trainset = My_SVHN('train')
        encoder = ResNet(BasicBlock, [2, 2, 2, 2])
        model = Projector(encoder, 512, 64)
        en_state = torch.load('./pretrain/Encoder/SVHN_E_checkpoint.tar',map_location='cpu')
        pr_state = torch.load('./pretrain/Projector/SVHN_P_checkpoint.tar',map_location='cpu')
        model.encoder.load_state_dict(en_state)
        model.projector.load_state_dict(pr_state)
        model = model.cuda()
        train_sampler = CategoriesSampler_v2(trainset.data, trainset.labels, 100, 5,
                                        args.train_way, args.shot + args.query, 0, 'svhn')
        train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                                num_workers=8, pin_memory=True)
        valset = My_SVHN('test')
        val_sampler = CategoriesSampler(valset.labels, 400,
                                        args.test_way, args.shot + args.query, 5)  
        val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                                num_workers=8, pin_memory=True)

    optimizer = torch.optim.SGD(model.projector.parameters(), lr=args.lr)
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    def save_model(name):
        torch.save(model.state_dict(), osp.join(args.save_path, name + '.pth'))
    
    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0

    timer = Timer()

    for epoch in range(1, args.max_epoch + 1):
        #lr_scheduler.step()

        model.train()

        tl = Averager()
        ta = Averager()

        for i, batch in enumerate(train_loader,1):
            data, _ = [_.cuda() for _ in batch]
            data = data.cuda()
            p = args.shot * args.train_way
            data_shot, data_query = data[:p], data[p:]

            _, proto = model(data_shot)
            proto = proto.reshape(args.shot, args.train_way, -1).mean(dim=0)
            
            label = torch.arange(args.train_way).repeat(args.query)
            label = label.type(torch.cuda.LongTensor).cuda()
            
            logits = euclidean_metric(model(data_query)[1], proto)
            pred = torch.argmax(logits, dim=1)
            
            loss = F.cross_entropy(logits, label)
            acc = cluster_acc(pred.cpu().numpy(), label.cpu().numpy())
            print('epoch {}, train {}/{}, loss={:.4f} acc={:.4f}'
                  .format(epoch, i, len(train_loader), loss.item(), acc))

            tl.add(loss.item())
            ta.add(acc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            proto = None; logits = None; loss = None

        tl = tl.item()
        ta = ta.item()

        model.eval()

        vl = Averager()
        va = Averager()

        for i, batch in enumerate(val_loader,1):
            proto = None; logits = None; loss = None
            data, _ = [_.cuda() for _ in batch]
            p = args.shot * args.test_way
            data_shot, data_query = data[:p], data[p:]

            _, proto = model(data_shot)
            proto = proto.reshape(args.shot, args.test_way, -1).mean(dim=0)
            
            label = torch.arange(args.test_way).repeat(args.query)
            label = label.type(torch.cuda.LongTensor)
            
            logits = euclidean_metric(model(data_query)[1], proto)
            pred = torch.argmax(logits, dim=1)
            
            loss = F.cross_entropy(logits, label)
            acc = cluster_acc(pred.cpu().numpy(), label.cpu().numpy())

            vl.add(loss.item())
            va.add(acc)

        vl = vl.item()
        va = va.item()
        print('epoch {}, val, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))
        
        true_label = np.array([])
        predict_label = np.array([])
        for (data, label) in test_loader:
            data, label = data.cuda(), label.cuda()
            logits = euclidean_metric(model(data)[1], proto)
            pred = torch.argmax(logits, dim=1)
            true_label = np.append(true_label, label.cpu().numpy())
            predict_label = np.append(predict_label, pred.cpu().numpy())
        acc = cluster_acc(true_label.astype(int), predict_label.astype(int))
        print('epoch {}, val on whole test set, acc={:.4f}'.format(epoch, acc))

        if va > trlog['max_acc']:
            trlog['max_acc'] = va
            save_model('max-acc')
             
        trlog['train_loss'].append(tl)
        trlog['train_acc'].append(ta)
        trlog['val_loss'].append(vl)
        trlog['val_acc'].append(va)

        torch.save(trlog, osp.join(args.save_path, 'trlog'))

        save_model('epoch-last')

        if epoch % args.save_epoch == 0:
            save_model('epoch-{}'.format(epoch))

        print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch)))
        
