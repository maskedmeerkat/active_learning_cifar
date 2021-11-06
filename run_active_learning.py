# Based on https://github.com/kuangliu/pytorch-cifar
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os
from models import *
from utils import (train_for_an_epoch, test_after_epoch, prepare_dataloaders, random_index_generation,
                   loss_and_probs_over_unlabeled, label_additional_data)
import copy
import torch
import random
import matplotlib.pyplot as plt
import time
import numpy as np
torch.manual_seed(119)
random.seed(119)
np.random.seed(119)
plt.close("all")


#===================================================================#
# Parameters
#===================================================================#
# dataset params
dataset_name = "mnist"
# dataset_name = "cifar"
data_root = os.path.join('<your dataset path>', dataset_name)
labelled_portion_steps = np.array([0.01, 0.05, 0.1, 0.2])  # initial portion will be randomly selected
epochs_per_step = np.array([100, 100, 100, 100])  # number of epochs per AL step

# model params
resume_ckpt_path = False # "./<your ckpt dir>/<your ckpt name>.pth"
ckpt_dir = './ckpts_mnist__loss_selection/'
net = ResNet18()
ckpt_model_name = "resnet18"

# training params
batch_size = 512
epochs = 200
lr = 0.1


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

last_labelled_amount = 0  # start with zero labelled data
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

for (current_labelled_portion, epochs) in zip(labelled_portion_steps, epochs_per_step):
    print("----------------------")
    print(f"Next AL Step: \033[92m{current_labelled_portion * 100}%\033[0m labelled data")

    if last_labelled_amount > 0:
        if resume_ckpt_path:
            print(f'==> Resuming from {resume_ckpt_path}..')
            checkpoint = torch.load(resume_ckpt_path)
            net.load_state_dict(checkpoint['net'])

        print("==> Label additional data..")
        time.sleep(0.01)
        # how many samples to label in addition to the randomly generated base line
        current_labeled_amount = int(loader_train_full.sampler.num_samples * current_labelled_portion)
        amount_still_to_label = current_labeled_amount - last_labelled_amount

        # get AL criterion
        losses, probs = loss_and_probs_over_unlabeled(net, loader_train_unlabeled)
        max_probs = probs.max(axis=1)
        randomized = np.arange(losses.shape[0])  # train data is already randomized with fixed seed

        # sort unlabelled data based on AL criterion
        # ToDo: NEED TO REPLACE WITH YOUR AL CRITERION!!!
        sorted_indices = np.argsort(losses)[::-1]  # take the ones with highest losses
        # sorted_indices = np.argsort(max_probs)
        # sorted_indices = np.argsort(randomized)
        indices_to_label = sorted_indices[:amount_still_to_label]
        indices_to_keep_unlabelled = sorted_indices[amount_still_to_label:]

        # label additional data based on AL criterion
        tmp = label_additional_data(train_set_labeled, train_set_unlabeled, indices_to_label, indices_to_keep_unlabelled, batch_size)
        loader_train_labeled, train_set_labeled, train_set_unlabeled = tmp
    else:
        # if checkpoint is provided
        if resume_ckpt_path:
            # get amount of labelled data the checkpoint has been trained with
            current_labelled_portion = int(resume_ckpt_path.split("/")[-1].split(".")[0].split("_")[2]) / 100

        print('==> Label initial data..')
        tmp = prepare_dataloaders(data_root, batch_size, current_labelled_portion, dataset_name)
        loader_train_full, loader_train_labeled, loader_train_unlabeled, loader_val, train_set_labeled, train_set_unlabeled = tmp
        current_labeled_amount = loader_train_labeled.sampler.num_samples

    # skip the initial training if a checkpoint is provided
    if not((last_labelled_amount == 0) and resume_ckpt_path):
        print("==> Start training..")
        time.sleep(0.01)
        best_acc = 0  # best test accuracy with current labelled data
        resume_ckpt_path = None  # reset the checkpoint
        for epoch in range(epochs):
            train_for_an_epoch(epoch, net, loader_train_labeled, optimizer, criterion)
            best_acc, resume_ckpt_path = test_after_epoch(epoch, net, loader_val, criterion, best_acc, ckpt_dir,
                                                          ckpt_model_name, current_labelled_portion, resume_ckpt_path)
            scheduler.step()

    # update the amount of labelled samples
    last_labelled_amount = current_labeled_amount
