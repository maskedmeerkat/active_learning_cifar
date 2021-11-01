# Based on https://github.com/kuangliu/pytorch-cifar
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os
from models import *
from utils import (train_for_an_epoch, test_after_epoch, prepare_dataloaders, random_index_generation,
                   acc_over_unlabeled, label_additional_data)
import copy
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(119)


#===================================================================#
# Parameters
#===================================================================#
# dataset params
data_root = '../_datasets/cifar/'
random_labelled_portion = 0.01
total_labelled_portion = 0.03

# model params
resume_ckpt_path = "./ckpts_random_selection/ckpt_resnet18_data_001_acc_44_2.pth"
ckpt_dir = './ckpts_acc_selection/'
net = ResNet18()
net_reload = ResNet18()
ckpt_model_name = "resnet18_data_003"

# training params
batch_size = 32
epochs = 200
lr = 0.02


#===================================================================#
# Prepare the Datasets
#===================================================================#
print('==> Preparing data..')
tmp = prepare_dataloaders(data_root, batch_size, random_labelled_portion)
loader_train_full, loader_train_labeled, loader_train_unlabeled, loader_val, train_set_labeled, train_set_unlabeled = tmp

# how many samples to label in addition to the randomly generated base line
total_num_to_labeled = int(loader_train_full.sampler.num_samples * total_labelled_portion)
num_of_still_to_label = total_num_to_labeled - loader_train_labeled.sampler.num_samples


#===================================================================#
# Prepare the Model
#===================================================================#
print('==> Building model..')
# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net_reload = net_reload.to(device)
if device == 'cuda':
    net_reload = torch.nn.DataParallel(net_reload)
    cudnn.benchmark = True

# Load checkpoint.
print('==> Load checkpoint..')
checkpoint = torch.load(os.path.join(resume_ckpt_path))
net_reload.load_state_dict(checkpoint['net'])


#===================================================================#
# Losses for each unlabelled Data
#===================================================================#
# compute loss for each unlabeled sample
# losses = acc_over_unlabeled(net_reload, loader_train_unlabeled, device)
losses = np.load("./docs/losses_on_unlabeled_ckpt_random_001.npy")

# sort the losses from highest to lowest
sorted_indices = np.argsort(losses)[::-1]
indices_to_sample = sorted_indices[:num_of_still_to_label]

# visualize the distributions before and after AL
plt.figure()
plt.hist(list(np.asarray(train_set_unlabeled.targets, dtype=np.int64)[indices_to_sample]))
plt.hist(train_set_labeled.targets)
plt.xticks(np.arange(10), train_set_labeled.classes)
plt.ylabel("num samples / class")
plt.legend(["class dist. after AL", "class dist. before AL"])

# label additional data
loader_train_labeled = label_additional_data(train_set_labeled, train_set_unlabeled, indices_to_sample, batch_size)


#===================================================================#
# Train the Model
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

# Load checkpoint.
print('==> Load checkpoint..')
checkpoint = torch.load(os.path.join(resume_ckpt_path))
net.load_state_dict(checkpoint['net'])

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

for epoch in range(start_epoch, start_epoch+epochs):
    train_for_an_epoch(epoch, net, loader_train_labeled, device, optimizer, criterion)
    best_acc = test_after_epoch(epoch, net, loader_val, device, criterion, best_acc, ckpt_dir, ckpt_model_name)
    scheduler.step()

