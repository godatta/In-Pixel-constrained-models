import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn as nn
from torchvision import datasets, transforms, models
import numpy as np
import argparse
#from misc import progress_bar
import pyvww
import pkbar
import torch
import time
import os

from MobileNetv2 import MobileNetV2
#from VGG1x1 import VGG
from VGG_large_ip import VGG
from Lenet5_caffe import LeNet_5_Caffe
from torchvision import transforms, utils


class weightConstraint(object):
    def __init__(self):
        pass
    
    def __call__(self,module):
        if hasattr(module,'weight'):
            #print("Entered")
            w=module.weight.data
            w=w.clamp(-3.0,3.0)
            module.weight.data=w


def train(model, args, train_loader, device, optimizer, criterion, constraints):
    print("train:")
    model.train()
    train_loss = 0
    train_correct = 0
    total = 0

    for batch_num, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        #model = nn.DataParallel(model)
        output = model(data)
        loss = criterion(output, target)
        loss.mean().backward()
        optimizer.step()
        #model.apply(constraints)
        train_loss += loss.item()
        prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced
        total += target.size(0)

        #for p in model.parameters():
        #    print("Gradients are printed below \n")
        #    print(p.grad)
        #print(list(model.parameters()))
        #for name, param in model.named_parameters():
        #    if param.requires_grad:
        #        print(param.grad)
        #        break


        train_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())
        if (batch_num % args.print_interval == 0):
            print('Batch:{}, % Acc:{}, total:{}'.format(batch_num, 100. * train_correct / total, total))
    return train_loss, train_correct / total

def test(model, epoch, args, test_loader, device, optimizer, criterion):
    print("test:")
    model.eval()
    test_loss = 0
    test_correct = 0
    total = 0

    with torch.no_grad():
        for batch_num, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            #model = nn.DataParallel(model)
            #model = model.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            prediction = torch.max(output, 1)
            total += target.size(0)
            test_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

            if (batch_num % args.print_interval == 0):
                print('Batch:{}, % Acc:{}, total:{}'.format(batch_num, 100. * test_correct / total, total))
    print('Epoch:{}, Average loss: {:.4f}, Top1 Acc: {}\n'.\
        format(epoch, test_loss, 100. * test_correct / total))
    return test_loss, 100. * test_correct / total

def main():
    parser = argparse.ArgumentParser(description="cifar-10 with PyTorch")
    parser.add_argument('--lr', default=0.02, type=float, help='learning rate')
    parser.add_argument('--epochs', default=100, type=int, help='number of epochs tp train for')
    parser.add_argument('--print_interval', default=50, type=int, help='number of epochs tp train for')
    parser.add_argument('--batch_size', default=64, type=int, help='training batch size')
    parser.add_argument('--test_batch_size', default=64, type=int, help='testing batch size')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--data', type=str, default='vww',
                        help='the dataset to be used for training [vww, mnist]')
    parser.add_argument('--img_res', type=int, default=224, help='the resolution of the input image')
    parser.add_argument('--model_type', type=str, default='mobilenetv2',
                         help = 'supported model types are [mobilenetv2, vgg11, vgg9, vgg5]')
    parser.add_argument('--optim', type=str, default='rmsprop',
                         help = 'supported model optimizers [rmsprop, adam]')
    parser.add_argument('--mode', type=str, default='test',
                         help = 'supported mode types [train, test]')
    parser.add_argument('--seed', type=int, default=1000, metavar='S', help='random seed')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)

    if (args.data == 'vww'):
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), \
            transforms.Resize(size=(args.img_res, args.img_res)), transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.Resize(size=(args.img_res, args.img_res)),\
             transforms.ToTensor()])

        train_set = pyvww.pytorch.VisualWakeWordsClassification(root="/home/gdatta/coco/all2014", \
            annFile = "/home/gdatta/visualwakewords/annotations/instances_train.json", transform=train_transform)
        train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
        test_set = pyvww.pytorch.VisualWakeWordsClassification(root="/home/gdatta/coco/all2014", \
                annFile="/home/gdatta/visualwakewords/annotations/instances_val.json", transform=test_transform)
        test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=args.test_batch_size, shuffle=False)
        num_classes = 2 
    
    elif (args.data == 'mnist'):
        normalize = transforms.Normalize((0.1307,), (0.3081,))
        transform = transform=transforms.Compose([transforms.ToTensor(),normalize])

        full_dataset = datasets.MNIST('/home/souvikku/_dataset', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('/home/souvikku/_dataset', train=False, transform=transform)

        dataset_size = len(full_dataset)
        indices = list(range(dataset_size))

        train_loader = torch.utils.data.DataLoader(
            full_dataset,
            args.batch_size,
            num_workers=8,
            pin_memory=True, shuffle=True)

        print('Train loader length', len(train_loader))

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            args.test_batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=True)
        num_classes = 10 

    
    if args.model_type == 'mobilenetv2':
        constraints=weightConstraint()
        model = MobileNetV2(num_classes=num_classes).to(device)
        print("all good")
        #model.apply(constraints)
    elif args.model_type == 'vgg11':
        model = VGG('VGG11', num_classes=num_classes)
    elif args.model_type == 'vgg9':
        model = VGG('VGG9', num_classes=num_classes)
    elif args.model_type == 'lenet5_caffe':
        model = LeNet_5_Caffe()

    optimizer = None
    if args.optim == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr = args.lr)
    elif args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr = args.lr)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[35, 45], gamma=0.2) # for vww
    criterion = nn.CrossEntropyLoss().to(device)

    
        
    if args.mode == "test":
        state = torch.load("mobilenetv2_vww_rmsprop_inpRes560_epochs50_testAcc_89.25424990693635_seed1000_custom_conv.pt")
        missing_keys, unexpected_keys = model.load_state_dict(state, strict=False)
        epoch = 1
        test_loss, test_acc = test(model, epoch, args, test_loader, device, optimizer, criterion)
        
    else:
        best_testAcc = [0]
        train_acc_list = []
        trloss = []
        
        if not args.train_from_scratch:
            state = torch.load("mobilenetv2_vww_rmsprop_inpRes560_epochs50_testAcc_89.25424990693635_seed1000_custom_conv.pt")
            missing_keys, unexpected_keys = model.load_state_dict(state, strict=False)
    
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            scheduler.step(epoch)
            train_loss, train_acc = train(model, args, train_loader, device, optimizer, criterion, constraints)
            train_acc_list.append(train_acc)
            trloss.append(train_loss)
            
        #model.apply(constraints)
            print('Current learning rate: {0}. Time taken for epoch: {1:.2f} seconds.\n'.\
                format(optimizer.param_groups[0]['lr'], time.time() - t0))
            test_loss, test_acc = test(model, epoch, args, test_loader, device, optimizer, criterion)
            print("Finishing epoch: {}". format(epoch))

        


            if test_acc > max(best_testAcc):
                print("\n>_ Got better accuracy, saving model with accuracy {:.3f}% now...\n". format(test_acc))
                torch.save(model.state_dict(), "{}_{}_{}_inpRes{}_epochs{}_testAcc_{}_seed{}_custom_conv_constrained.pt".\
                    format(args.model_type, args.data, args.optim, args.img_res, args.epochs, test_acc, args.seed))
                print("\n>_ Deleting previous model file with accuracy {:.3f}% now...\n".format(max(best_testAcc)))
                if len(best_testAcc) > 1:
                    os.remove("{}_{}_{}_inpRes{}_epochs{}_testAcc_{}_seed{}_custom_conv_constrained.pt".\
                        format(args.model_type, args.data, args.optim, args.img_res, args.epochs, max(best_testAcc), args.seed))
            best_testAcc.append(test_acc)

    

    
        print('End iteration.\n')
        with open('{}_{}_{}_inpRes{}_epochs{}_testAccList_seed{}_custom_conv_constrained.txt'.\
            format(args.model_type, args.data, args.optim, args.img_res, args.epochs, args.seed), 'w') as f:
            for item in best_testAcc:
                f.write("%s\n" % item)
        with open('{}_{}_{}_inpRes{}_epochs{}_trainAccList_seed{}_custom_conv_constrained.txt'.\
            format(args.model_type, args.data, args.optim, args.img_res, args.epochs, args.seed), 'w') as f:
            for item in train_acc_list:
                f.write("%s\n" % item)
        with open('{}_{}_{}_inpRes{}_epochs{}_trainLoss_seed{}_custom_conv_constrained.txt'.\
            format(args.model_type, args.data, args.optim, args.img_res, args.epochs, args.seed), 'w') as f:
            for item in trloss:
                f.write("%s\n" % item)


if __name__ == '__main__':
    main()