# Modified by Hao Di on 2025-06-17
# Based on original work licensed under the Apache License, Version 2.0

import os
import pickle
import string

import torch
from torchvision.datasets import CIFAR10, CIFAR100, EMNIST, MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


import numpy as np
from PIL import Image


class TabularDataset(Dataset):
    """
    Constructs a torch.utils.Dataset object from a pickle file;
    expects pickle file stores tuples of the form (x, y) where x is vector and y is a scalar

    Attributes
    ----------
    data: iterable of tuples (x, y)

    Methods
    -------
    __init__
    __len__
    __getitem__
    """

    def __init__(self, path):
        """
        :param path: path to .pkl file
        """
        with open(path, "rb") as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.int64), idx

class SubHAR(Dataset):
    def __init__(self, path, data, targets):
        with open(path, "rb") as f:
            self.indices = pickle.load(f)
        self.data = torch.tensor(data[self.indices], dtype=torch.float32)
        self.targets = torch.tensor(targets[self.indices], dtype=torch.int64)
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, index):
        return self.data[index], self.targets[index], index

class SubFEMNIST(Dataset):
    """
    Constructs a subset of FEMNIST dataset corresponding to one client;
    Initialized with the path to a `.pt` file;
    `.pt` file is expected to hold a tuple of tensors (data, targets) storing the images and there corresponding labels.

    Attributes
    ----------
    transform
    data: iterable of integers
    targets

    Methods
    -------
    __init__
    __len__
    __getitem__
    """
    def __init__(self, path):
        self.transform = Compose([
            ToTensor(),
            Normalize((0.1307,), (0.3081,))
        ])

        self.data, self.targets = torch.load(path)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        img = np.uint8(img.numpy() * 255)
        img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return img, target, index


class SubEMNIST(Dataset):
    """
    Constructs a subset of EMNIST dataset from a pickle file;
    expects pickle file to store list of indices

    Attributes
    ----------
    indices: iterable of integers
    transform
    data
    targets

    Methods
    -------
    __init__
    __len__
    __getitem__
    """

    def __init__(self, path, emnist_data=None, emnist_targets=None, transform=None):
        """
        :param path: path to .pkl file; expected to store list of indices
        :param emnist_data: EMNIST dataset inputs
        :param emnist_targets: EMNIST dataset labels
        :param transform:
        """
        with open(path, "rb") as f:
            self.indices = pickle.load(f)

        if transform is None:
            self.transform =\
                Compose([
                    ToTensor(),
                    Normalize((0.1307,), (0.3081,))
                ])

        if emnist_data is None or emnist_targets is None:
            self.data, self.targets = get_emnist()
        else:
            self.data, self.targets = emnist_data, emnist_targets

        self.data = self.data[self.indices]
        self.targets = self.targets[self.indices]

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return img, target, index

class SubMNIST(Dataset):

    def __init__(self, path, data, targets) -> None:
        with open(path, 'rb') as f:
            self.indices = pickle.load(f)
        
        self.transform = Compose([
            ToTensor(),
            Normalize((0.1307,), (0.3081,))
        ])

        self.data = data[self.indices]
        self.targets = targets[self.indices]
    
    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        img = Image.fromarray(img.numpy(), mode='L')

        img = self.transform(img)
        
        return img, target, index


class SubCIFAR10(Dataset):
    """
    Constructs a subset of CIFAR10 dataset from a pickle file;
    expects pickle file to store list of indices

    Attributes
    ----------
    indices: iterable of integers
    transform
    data
    targets

    Methods
    -------
    __init__
    __len__
    __getitem__
    """
    def __init__(self, path, cifar10_data=None, cifar10_targets=None, transform=None):
        """
        :param path: path to .pkl file; expected to store list of indices
        :param cifar10_data: Cifar-10 dataset inputs stored as torch.tensor
        :param cifar10_targets: Cifar-10 dataset labels stored as torch.tensor
        :param transform:
        """
        with open(path, "rb") as f:
            self.indices = pickle.load(f)

        if transform is None:
            self.transform = \
                Compose([
                    ToTensor(),
                    Normalize(
                        (0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010)
                    )
                ])

        if cifar10_data is None or cifar10_targets is None:
            self.data, self.targets = get_cifar10()
        else:
            self.data, self.targets = cifar10_data, cifar10_targets

        self.data = self.data[self.indices]
        self.targets = self.targets[self.indices]

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img.numpy())

        if self.transform is not None:
            img = self.transform(img)

        target = target

        return img, target, index


class SubCIFAR100(Dataset):
    """
    Constructs a subset of CIFAR100 dataset from a pickle file;
    expects pickle file to store list of indices

    Attributes
    ----------
    indices: iterable of integers
    transform
    data
    targets

    Methods
    -------
    __init__
    __len__
    __getitem__
    """
    def __init__(self, path, cifar100_data=None, cifar100_targets=None, transform=None):
        """
        :param path: path to .pkl file; expected to store list of indices:
        :param cifar100_data: CIFAR-100 dataset inputs
        :param cifar100_targets: CIFAR-100 dataset labels
        :param transform:
        """
        with open(path, "rb") as f:
            self.indices = pickle.load(f)

        if transform is None:
            self.transform = \
                Compose([
                    ToTensor(),
                    Normalize(
                        (0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010)
                    )
                ])

        if cifar100_data is None or cifar100_targets is None:
            self.data, self.targets = get_cifar100()

        else:
            self.data, self.targets = cifar100_data, cifar100_targets

        self.data = self.data[self.indices]
        self.targets = self.targets[self.indices]

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img.numpy())

        if self.transform is not None:
            img = self.transform(img)

        target = target

        return img, target, index


class CharacterDataset(Dataset):
    def __init__(self, file_path, chunk_len):
        """
        Dataset for next character prediction, each sample represents an input sequence of characters
         and a target sequence of characters representing to next sequence of the input
        :param file_path: path to .txt file containing the training corpus
        :param chunk_len: (int) the length of the input and target sequences
        """
        self.all_characters = string.printable
        self.vocab_size = len(self.all_characters)
        self.n_characters = len(self.all_characters)
        self.chunk_len = chunk_len

        with open(file_path, 'r') as f:
            self.text = f.read()

        self.tokenized_text = torch.zeros(len(self.text), dtype=torch.long)

        self.inputs = torch.zeros(self.__len__(), self.chunk_len, dtype=torch.long)
        self.targets = torch.zeros(self.__len__(), self.chunk_len, dtype=torch.long)

        self.__build_mapping()
        self.__tokenize()
        self.__preprocess_data()

    def __tokenize(self):
        for ii, char in enumerate(self.text):
            self.tokenized_text[ii] = self.char2idx[char]

    def __build_mapping(self):
        self.char2idx = dict()
        for ii, char in enumerate(self.all_characters):
            self.char2idx[char] = ii

    def __preprocess_data(self):
        for idx in range(self.__len__()):
            self.inputs[idx] = self.tokenized_text[idx:idx+self.chunk_len]
            self.targets[idx] = self.tokenized_text[idx+1:idx+self.chunk_len+1]

    def __len__(self):
        return max(0, len(self.text) - self.chunk_len)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx], idx


class DigitsDataset(Dataset):
    def __init__(self, data_path, channels, percent=0.1, filename=None, train=True, transform=None):
        if filename is None:
            if train:
                if percent >= 0.1:
                    for part in range(int(percent*10)):
                        if part == 0:
                            self.images, self.labels = np.load(os.path.join(data_path, 'partitions/train_part{}.pkl'.format(part)), allow_pickle=True)
                        else:
                            images, labels = np.load(os.path.join(data_path, 'partitions/train_part{}.pkl'.format(part)), allow_pickle=True)
                            self.images = np.concatenate([self.images,images], axis=0)
                            self.labels = np.concatenate([self.labels,labels], axis=0)
                else:
                    self.images, self.labels = np.load(os.path.join(data_path, 'partitions/train_part0.pkl'), allow_pickle=True)
                    data_len = int(self.images.shape[0] * percent*10)
                    self.images = self.images[:data_len]
                    self.labels = self.labels[:data_len]
            else:
                self.images, self.labels = np.load(os.path.join(data_path, 'test.pkl'), allow_pickle=True)
        else:
            self.images, self.labels = np.load(os.path.join(data_path, filename), allow_pickle=True)

        self.transform = transform
        self.channels = channels
        self.labels = self.labels.astype(np.long).squeeze()

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.channels == 1:
            image = Image.fromarray(image, mode='L')
        elif self.channels == 3:
            image = Image.fromarray(image, mode='RGB')
        else:
            raise ValueError("{} channel is not allowed.".format(self.channels))

        if self.transform is not None:
            image = self.transform(image)

        return image, label, idx


def get_covariate(batch_size):
    # Prepare data
    transform_mnist = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_svhn = transforms.Compose([
            transforms.Resize([28,28]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_usps = transforms.Compose([
            transforms.Resize([28,28]),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_synth = transforms.Compose([
            transforms.Resize([28,28]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_mnistm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    # MNIST
    mnist_trainset     = DigitsDataset(data_path="./data/covariate/MNIST", channels=1, percent=0.1, train=True,  transform=transform_mnist)
    mnist_valset     = DigitsDataset(data_path="./data/covariate/MNIST", channels=1, percent=0.1, train=True,  transform=transform_mnist)
    mnist_testset      = DigitsDataset(data_path="./data/covariate/MNIST", channels=1, percent=0.1, train=False, transform=transform_mnist)

    # SVHN
    svhn_trainset      = DigitsDataset(data_path='./data/covariate/SVHN', channels=3, percent=0.1,  train=True,  transform=transform_svhn)
    svhn_valset      = DigitsDataset(data_path='./data/covariate/SVHN', channels=3, percent=0.1,  train=True,  transform=transform_svhn)
    svhn_testset       = DigitsDataset(data_path='./data/covariate/SVHN', channels=3, percent=0.1,  train=False, transform=transform_svhn)

    # USPS
    usps_trainset      = DigitsDataset(data_path='./data/covariate/USPS', channels=1, percent=0.1,  train=True,  transform=transform_usps)
    usps_valset      = DigitsDataset(data_path='./data/covariate/USPS', channels=1, percent=0.1,  train=True,  transform=transform_usps)
    usps_testset       = DigitsDataset(data_path='./data/covariate/USPS', channels=1, percent=0.1,  train=False, transform=transform_usps)

    # Synth Digits
    synth_trainset     = DigitsDataset(data_path='./data/covariate/SynthDigits/', channels=3, percent=0.1,  train=True,  transform=transform_synth)
    synth_valset     = DigitsDataset(data_path='./data/covariate/SynthDigits/', channels=3, percent=0.1,  train=True,  transform=transform_synth)
    synth_testset      = DigitsDataset(data_path='./data/covariate/SynthDigits/', channels=3, percent=0.1,  train=False, transform=transform_synth)

    # MNIST-M
    mnistm_trainset     = DigitsDataset(data_path='./data/covariate/MNIST_M/', channels=3, percent=0.1,  train=True,  transform=transform_mnistm)
    mnistm_valset     = DigitsDataset(data_path='./data/covariate/MNIST_M/', channels=3, percent=0.1,  train=True,  transform=transform_mnistm)
    mnistm_testset      = DigitsDataset(data_path='./data/covariate/MNIST_M/', channels=3, percent=0.1,  train=False, transform=transform_mnistm)

    mnist_train_loader = DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True)
    mnist_val_loader = DataLoader(mnist_valset, batch_size=batch_size, shuffle=False)
    mnist_test_loader  = DataLoader(mnist_testset, batch_size=batch_size, shuffle=False)
    svhn_train_loader = DataLoader(svhn_trainset, batch_size=batch_size,  shuffle=True)
    svhn_val_loader = DataLoader(svhn_valset, batch_size=batch_size,  shuffle=False)
    svhn_test_loader = DataLoader(svhn_testset, batch_size=batch_size, shuffle=False)
    usps_train_loader = DataLoader(usps_trainset, batch_size=batch_size,  shuffle=True)
    usps_val_loader = DataLoader(usps_valset, batch_size=batch_size,  shuffle=False)
    usps_test_loader = DataLoader(usps_testset, batch_size=batch_size, shuffle=False)
    synth_train_loader = DataLoader(synth_trainset, batch_size=batch_size,  shuffle=True)
    synth_val_loader = DataLoader(synth_valset, batch_size=batch_size,  shuffle=False)
    synth_test_loader = DataLoader(synth_testset, batch_size=batch_size, shuffle=False)
    mnistm_train_loader = DataLoader(mnistm_trainset, batch_size=batch_size,  shuffle=True)
    mnistm_val_loader = DataLoader(mnistm_valset, batch_size=batch_size,  shuffle=False)
    mnistm_test_loader = DataLoader(mnistm_testset, batch_size=batch_size, shuffle=False)

    train_loaders = [mnist_train_loader, svhn_train_loader, usps_train_loader, synth_train_loader, mnistm_train_loader]
    val_loaders = [mnist_val_loader, svhn_val_loader, usps_val_loader, synth_val_loader, mnistm_val_loader]
    test_loaders  = [mnist_test_loader, svhn_test_loader, usps_test_loader, synth_test_loader, mnistm_test_loader]

    return train_loaders, val_loaders, test_loaders



def get_emnist():
    """
    gets full (both train and test) EMNIST dataset inputs and labels;
    the dataset should be first downloaded (see data/emnist/README.md)
    :return:
        emnist_data, emnist_targets
    """
    emnist_path = os.path.join("data", "emnist", "raw_data")
    assert os.path.isdir(emnist_path), "Download EMNIST dataset!!"

    emnist_train =\
        EMNIST(
            root=emnist_path,
            split="byclass",
            download=True,
            train=True
        )

    emnist_test =\
        EMNIST(
            root=emnist_path,
            split="byclass",
            download=True,
            train=False
        )

    emnist_data =\
        torch.cat([
            emnist_train.data,
            emnist_test.data
        ])

    emnist_targets =\
        torch.cat([
            emnist_train.targets,
            emnist_test.targets
        ])

    return emnist_data, emnist_targets

def get_har():
    har_path = os.path.join("data", "har", "raw_data", "UCI HAR Dataset")
    assert os.path.isdir(har_path), "Download HAR dataset!!"

    har_train, har_train_target = np.loadtxt(os.path.join(har_path, 'train', 'X_train.txt')), np.loadtxt(os.path.join(har_path, 'train', 'y_train.txt'))
    har_test, har_test_target = np.loadtxt(os.path.join(har_path, 'test', 'X_test.txt')), np.loadtxt(os.path.join(har_path, 'test', 'y_test.txt'))

    har_data = np.concatenate([har_train, har_test], dtype=np.float32)
    har_target = np.concatenate([har_train_target, har_test_target]).astype(np.int32)

    # mapping people activities into a binary label: walking or static.
    walking_state = [1, 2, 3]
    static_state = [4, 5, 6]
    for state in walking_state:
        idx = np.where(har_target == state)[0]
        har_target[idx] = [0] * len(idx)
    for state in static_state:
        idx = np.where(har_target == state)[0]
        har_target[idx] = [1] * len(idx)

    return har_data, har_target


def get_mnist():
    mnist_path = os.path.join("data", "mnist", "raw_data")
    assert os.path.isdir(mnist_path), "Download MNIST dataset!!"

    mnist_train = MNIST(
        root= mnist_path,
        download=True,
        train=True
    )

    mnist_test = MNIST(
        root=mnist_path,
        download=True,
        train=False
        )

    mnist_data =\
        torch.cat([
            mnist_train.data,
            mnist_test.data
        ])

    mnist_targets =\
        torch.cat([
            mnist_train.targets,
            mnist_test.targets
        ])

    return mnist_data, mnist_targets



def get_cifar10():
    """
    gets full (both train and test) CIFAR10 dataset inputs and labels;
    the dataset should be first downloaded (see data/emnist/README.md)
    :return:
        cifar10_data, cifar10_targets
    """
    cifar10_path = os.path.join("data", "cifar10", "raw_data")
    assert os.path.isdir(cifar10_path), "Download cifar10 dataset!!"

    cifar10_train =\
        CIFAR10(
            root=cifar10_path,
            train=True, download=False
        )

    cifar10_test =\
        CIFAR10(
            root=cifar10_path,
            train=False,
            download=False)

    cifar10_data = \
        torch.cat([
            torch.tensor(cifar10_train.data),
            torch.tensor(cifar10_test.data)
        ])

    cifar10_targets = \
        torch.cat([
            torch.tensor(cifar10_train.targets),
            torch.tensor(cifar10_test.targets)
        ])

    return cifar10_data, cifar10_targets


def get_cifar100():
    """
    gets full (both train and test) CIFAR100 dataset inputs and labels;
    the dataset should be first downloaded (see data/cifar100/README.md)
    :return:
        cifar100_data, cifar100_targets
    """
    cifar100_path = os.path.join("data", "cifar100", "raw_data")
    assert os.path.isdir(cifar100_path), "Download cifar10 dataset!!"

    cifar100_train =\
        CIFAR100(
            root=cifar100_path,
            train=True, download=False
        )

    cifar100_test =\
        CIFAR100(
            root=cifar100_path,
            train=False,
            download=False)

    cifar100_data = \
        torch.cat([
            torch.tensor(cifar100_train.data),
            torch.tensor(cifar100_test.data)
        ])

    cifar100_targets = \
        torch.cat([
            torch.tensor(cifar100_train.targets),
            torch.tensor(cifar100_test.targets)
        ])

    return cifar100_data, cifar100_targets
