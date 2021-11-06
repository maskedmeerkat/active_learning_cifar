import torch
import torch.nn as nn
import torch.nn.init as init
import os
import glob
import torchvision.transforms as transforms
from tqdm import tqdm
import torchvision
import copy
import numpy as np
from PIL import Image


class MNIST(torchvision.datasets.MNIST):
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if isinstance(img, np.ndarray):
            img = np.tile(img[:, :, np.newaxis], (1, 1, 3))
        else:
            img = np.tile(img.numpy()[:, :, np.newaxis], (1, 1, 3))
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def get_mean_and_std(dataset):
    """ Compute the mean and std value of dataset. """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    """ Init layer parameters. """
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


def train_for_an_epoch(epoch, net, train_loader, optimizer, criterion):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # setup progressbar
    train_loader_with_progbar = tqdm(train_loader, unit="batch")
    train_loader_with_progbar.set_description(f"Epoch {epoch}")

    # setup for next epoch
    net.train()
    running_loss = 0
    running_accuracy = 0
    total_num_samples = 0
    for i_batch, (inputs, targets) in enumerate(train_loader_with_progbar):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        # perform one optimization step
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # compute metrics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        running_accuracy += 100. * predicted.eq(targets).sum().item()
        total_num_samples += targets.size(0)

        # update progressbar
        train_loader_with_progbar.set_postfix(
            train_loss=running_loss / (i_batch + 1),
            train_accuracy=running_accuracy / total_num_samples
        )


def test_after_epoch(epoch, net, val_loader, criterion, best_acc, ckpt_dir, ckpt_model_name,
                     current_labelled_portion, resume_ckpt_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # setup progressbar
    val_loader_with_progbar = tqdm(val_loader, unit="batch")
    val_loader_with_progbar.set_description(f"      {epoch}")

    # setup for next epoch
    net.eval()
    running_loss = 0
    running_correct = 0
    total_num_samples = 0
    with torch.no_grad():
        for i_batch, (inputs, targets) in enumerate(val_loader_with_progbar):
            # perform inference
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            # compute metrics
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            running_correct += 100. * predicted.eq(targets).sum().item()
            total_num_samples += targets.size(0)

            # update progressbar
            val_loader_with_progbar.set_postfix(
                val_loss=running_loss / (i_batch + 1),
                val_accuracy=f"\033[94m{round(running_correct / total_num_samples, 2)}\033[0m"
            )

    # Save checkpoint.
    average_accuracy = running_correct / total_num_samples
    if average_accuracy > best_acc:
        print('Saving..')
        # create checkpoint directory, if it doesn't exist
        os.makedirs(ckpt_dir, exist_ok=True)
        # create a state to store
        state = {'net': net.state_dict(), 'acc': average_accuracy, 'epoch': epoch}
        # define a ckpt name
        accuracy_str = f"acc_{round(average_accuracy)}_{round(average_accuracy%1*10)}"
        data_portion_str = f"labelled_{int(current_labelled_portion*100):03}"
        resume_ckpt_path = os.path.join(ckpt_dir, f'ckpt_{data_portion_str}_{ckpt_model_name}_{accuracy_str}.pth')
        # remove all ckpts in current ckpt dir that used the same portion of labelled data
        [os.remove(filename) for filename in glob.glob(os.path.join(ckpt_dir, f'ckpt_{data_portion_str}_*'))]
        # store the new ckpt
        torch.save(state, resume_ckpt_path)
        # update the best accuracy
        best_acc = average_accuracy

    return best_acc, resume_ckpt_path


def loss_and_probs_over_unlabeled(net, val_loader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # setup progressbar
    val_loader_with_progbar = tqdm(val_loader, unit="batch")
    val_loader_with_progbar.set_description(f"Accuracy over unlabeled Data ")

    # setup for next epoch
    net.eval()
    batch_size = val_loader.batch_size
    losses = np.zeros(val_loader.sampler.num_samples)
    probs = np.zeros((val_loader.sampler.num_samples, len(val_loader.dataset.classes)))
    criterion = nn.CrossEntropyLoss(reduction='none')
    with torch.no_grad():
        for i_batch, (inputs, targets) in enumerate(val_loader_with_progbar):
            # perform inference
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            probs = outputs.cpu().detach().numpy()

            # compute metrics
            curr_losses = criterion(outputs, targets).cpu().detach().numpy()
            losses[i_batch*batch_size:(i_batch+1)*batch_size] = curr_losses
    return losses, probs


def prepare_dataloaders(data_root, batch_size, labelled_portion, dataset_name):
    # define the dataloader transformation for train & val
    trafo_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trafo_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # prepare the full train & test datasets
    assert not ("mnsit" in ["mnist", "cifar"]), "Dataset name must be in ['mnist', 'cifar']"
    if dataset_name == "mnist":
        train_set_full = MNIST(root=data_root, train=True, download=True, transform=trafo_train)
        val_set = MNIST(root=data_root, train=False, download=True, transform=trafo_val)
    else:
        train_set_full = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=trafo_train)
        val_set = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=trafo_val)

    # generate random indices to split the training set into a labeled and unlabeled set
    random_indices, num_labeled_train_samples = random_index_generation(train_set_full, labelled_portion)

    # prepare a labelled and unlabelled part of the train dataset
    train_set_labeled = copy.copy(train_set_full)
    train_set_unlabeled = copy.copy(train_set_full)
    train_set_labeled.data = train_set_full.data[random_indices[:num_labeled_train_samples], ...]
    train_set_labeled.targets = list(np.asarray(train_set_full.targets, dtype=np.int64)[random_indices[:num_labeled_train_samples]])
    train_set_unlabeled.data = train_set_full.data[random_indices[num_labeled_train_samples:], ...]
    train_set_unlabeled.targets = list(np.asarray(train_set_full.targets, dtype=np.int64)[random_indices[num_labeled_train_samples:]])

    # create data_loaders for the datasets
    loader_train_labeled = torch.utils.data.DataLoader(train_set_labeled,
                                                       batch_size=batch_size, shuffle=True, num_workers=0)
    loader_train_unlabeled = torch.utils.data.DataLoader(train_set_unlabeled,
                                                         batch_size=batch_size, shuffle=True, num_workers=0)
    loader_train_full = torch.utils.data.DataLoader(train_set_full,
                                                    batch_size=batch_size, shuffle=True, num_workers=0)
    loader_val = torch.utils.data.DataLoader(val_set,
                                           batch_size=batch_size, shuffle=False, num_workers=0)

    return loader_train_full, loader_train_labeled, loader_train_unlabeled, loader_val, train_set_labeled, train_set_unlabeled


def random_index_generation(train_set_full, labelled_portion):
    num_train_samples = len(train_set_full.data)
    random_indices = np.random.choice(num_train_samples, num_train_samples, replace=False)
    num_labeled_samples = int(num_train_samples * labelled_portion)
    return random_indices, num_labeled_samples


def label_additional_data(train_set_labeled, train_set_unlabeled,
                          indices_to_label, indices_to_keep_unlabelled, batch_size):
    indices_to_label = np.sort(indices_to_label)
    indices_to_keep_unlabelled = np.sort(indices_to_keep_unlabelled)
    # append data and labels to the labelled train set
    train_set_labeled.data = np.append(train_set_labeled.data,
                                       train_set_unlabeled.data[indices_to_label, ...], axis=0)
    train_set_labeled.targets += list(np.asarray(train_set_unlabeled.targets,
                                                 dtype=np.int64)[indices_to_label])

    # remove the same data and labels from the unlabelled train set
    train_set_unlabeled.data = np.append(train_set_unlabeled.data,
                                         train_set_unlabeled.data[indices_to_keep_unlabelled, ...], axis=0)
    train_set_unlabeled.targets += list(np.asarray(train_set_unlabeled.targets,
                                                   dtype=np.int64)[indices_to_keep_unlabelled])

    # create a data loader for the labelled train set
    loader_train_labeled = torch.utils.data.DataLoader(train_set_labeled,
                                                       batch_size=batch_size, shuffle=True, num_workers=0)

    return loader_train_labeled, train_set_labeled, train_set_unlabeled

