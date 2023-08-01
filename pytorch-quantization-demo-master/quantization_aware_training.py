from model import *

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
import os.path as osp


def quantize_aware_training(model, device, train_loader, optimizer, epoch):
    lossLayer = torch.nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader, 1):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model.quantize_forward(data)
        loss = lossLayer(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 50 == 0:
            print('Quantize Aware Training Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), loss.item()
            ))


def full_inference(model, test_loader):
    correct = 0
    for i, (data, target) in enumerate(test_loader, 1):
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    print('\nTest set: Full Model Accuracy: {:.2f}%\n'.format(100. * correct / len(test_loader.dataset)))


def quantize_inference(model, test_loader):
    correct = 0
    for i, (data, target) in enumerate(test_loader, 1):
        data, target = data.to(device), target.to(device)
        output = model.quantize_inference(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    print('\nTest set: Quant Model Accuracy: {:.2f}%\n'.format(100. * correct / len(test_loader.dataset)))


if __name__ == "__main__":
    batch_size = 128
    test_batch_size = 100
    seed = 1
    epochs = 3
    lr = 0.001
    momentum = 0.9
    using_bn = True
    load_quant_model_file = None
    #load_quant_model_file = "ckpt/cifar10_cnnbn_qat.pt"

    torch.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
    ])

    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data', train=True, download=True, transform = transform_train),
        batch_size=batch_size, shuffle=True, num_workers=2)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data', train=False, transform = transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=2)

    if using_bn:
        model = NetBN()
        model.load_state_dict(torch.load('ckpt/cifar10_cnnbn.pt', map_location='cpu'))
        save_file = "ckpt/cifar10_cnnbn_qat.pt"
    else:
        model = Net()
        model.load_state_dict(torch.load('ckpt/cifar10_cnn.pt', map_location='cpu'))
        save_file = "ckpt/cifar10_cnn_qat.pt"
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    model.eval()
    
    full_inference(model, test_loader)

    num_bits = 8
    model.quantize(num_bits=num_bits)
    print('Quantization bit: %d' % num_bits)

    if load_quant_model_file is not None:
        model.load_state_dict(torch.load(load_quant_model_file))
        print("Successfully load quantized model %s" % load_quant_model_file)

    model.train()

    for epoch in range(1, epochs + 1):
        quantize_aware_training(model, device, train_loader, optimizer, epoch)

    model.eval()
    torch.save(model.state_dict(), save_file)

    model.freeze()

    quantize_inference(model, test_loader)

    



    
