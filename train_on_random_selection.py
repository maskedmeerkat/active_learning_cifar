# Based on https://github.com/kuangliu/pytorch-cifar
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os
from models import *
from utils import train_for_an_epoch, test_after_epoch, prepare_dataloaders, random_index_generation
import copy
import numpy as np
np.random.seed(119)


#===================================================================#
# Parameters
#===================================================================#
# dataset params
data_root = '../_datasets/cifar/'
labelled_portion = 0.01

# model params
resume_ckpt = False  # set to false, if you don't want to resume
ckpt_dir = './ckpts_random/'
net = ResNet18()
ckpt_model_name = "resnet18_data_001"

# training params
batch_size = 32
epochs = 200
lr = 0.02


#===================================================================#
# Prepare the Datasets
#===================================================================#
print('==> Preparing data..')
tmp = prepare_dataloaders(data_root, batch_size, labelled_portion)
loader_train_full, loader_train_labeled, loader_train_unlabeled, loader_val, train_set_labeled, train_set_unlabeled = tmp


#===================================================================#
# Prepare the Model
#===================================================================#
print('==> Building model..')
# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
if resume_ckpt:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(ckpt_dir), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(ckpt_dir, resume_ckpt))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


#===================================================================#
# Train the Model
#===================================================================#
for epoch in range(start_epoch, start_epoch+epochs):
    train_for_an_epoch(epoch, net, loader_train_labeled, device, optimizer, criterion)
    best_acc = test_after_epoch(epoch, net, loader_val, device, criterion, best_acc, ckpt_dir, ckpt_model_name)
    scheduler.step()

