import os
import random
import argparse
import pickle

import numpy as np

from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import ConcatDataset

from sklearn.utils import shuffle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


from typing import List
from collections import Counter, OrderedDict


N_CLASSES = 10
N_COMPONENTS = 3
SEED = 12345
RAW_DATA_PATH = "raw_data/"
PATH = "all_data/"

def save_data(l, path_):
    with open(path_, 'wb') as f:
        pickle.dump(l, f)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--n_tasks',
        help='number of tasks/clients;',
        type=int,
        required=True)
    parser.add_argument(
        '--n_components',
        help='number of components/clusters;',
        type=int,
        default=N_COMPONENTS
    )
    parser.add_argument(
        '--s_frac',
        help='fraction of the dataset to be used; default: 1.0;',
        type=float,
        default=1.0
    )
    parser.add_argument(
        '--tr_frac',
        help='fraction in training set; default: 0.8;',
        type=float,
        default=0.8
    )
    parser.add_argument(
        '--val_frac',
        help='fraction of validation set (from train set); default: 0.0;',
        type=float,
        default=0.0
    )
    parser.add_argument(
        '--test_tasks_frac',
        help='fraction of tasks / clients not participating to the training; default is 0.0',
        type=float,
        default=0.0
    )
    parser.add_argument(
        '--seed',
        help='seed for the random processes;',
        type=int,
        default=SEED
    )
    parser.add_argument(
        '--alpha',
        help='the parameter for the Dirichlet distribution;',
        type=float,
        default=1.0
    )

    return parser.parse_args()

def rejust_cosine(matrix) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    matrix -= np.mean(matrix, axis=0)
    res = cosine_similarity(matrix)
    return 0.5 + 0.5 * res

def save_metadata(path_, datas, data_names):
    metadata = dict()
    for name, data in zip(data_names, datas):
        metadata[name] = data
    with open(path_, 'wb') as f:
        pickle.dump(metadata, f)

def domain_labels(n_components:int, seed=1234)->List[List[int]]:
    rng = random.Random(seed)
    all_labels = list(range(10))
    vehicle, animal = [0, 1,8,9], [2,3,4,5,6,7]
    if n_components == 2:
        return [vehicle, animal]
    elif n_components == 3:
        return [[1, 9], [0, 8], animal]
    elif n_components == 4:
        sub_animal = rng.sample(animal, 3)
        return [[0, 8], [1, 9], sub_animal, list(set(animal) - set(sub_animal))]


def main():
    args = parse_args()

    transform = Compose([
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    dataset =\
        ConcatDataset([
            CIFAR10(root=RAW_DATA_PATH, download=True, train=True, transform=transform),
            CIFAR10(root=RAW_DATA_PATH, download=True, train=False, transform=transform)
        ])

    n_clients = args.n_tasks
    n_domain = args.n_components
    domains = domain_labels(n_domain, args.seed)

    rng = random.Random(args.seed)
    all_labels = np.array([data[1] for data in dataset])

    source_indices = []

    for i, domain in enumerate(domains):
        domain_idx = []
        # collect all labels' indexes in the specific domain
        for label in domain:
            label_idx = np.where(all_labels == label)[0]
            domain_idx.extend(label_idx)
        rng.shuffle(domain_idx)
        source_indices.append(domain_idx)
    
    np.random.seed(args.seed + 1)
    mixture_weights = np.random.dirichlet([args.alpha] * n_clients, N_COMPONENTS)
    mixture_weights = mixture_weights / mixture_weights.sum(axis=1, keepdims=True)

    save_metadata(os.path.join(PATH, "meta.pkl"), [mixture_weights], ["mixture_weights"])

    os.makedirs(os.path.join(PATH, "train"), exist_ok=True)
    os.makedirs(os.path.join(PATH, "test"), exist_ok=True)

    clients_idcs = [[] for _ in range(n_clients)]
    for k_idcs, fracs in zip(source_indices, mixture_weights):
        for i, idcs in enumerate(np.split(k_idcs,
                                            (np.cumsum(fracs)[:-1]*len(k_idcs)).
                                            astype(int))):
            clients_idcs[i] += [idcs]
            
    clients_idcs = [np.concatenate(idcs) for idcs in clients_idcs]

    clients_labels = np.zeros((n_clients, N_CLASSES))

    train_clients_indices, test_clients_indices = clients_idcs, []

    for mode, clients_indices in [('train', train_clients_indices), ('test', test_clients_indices)]:
        for client_id, indices in enumerate(clients_indices):
            if len(indices) == 0:
                continue

            labels_cnt = Counter([dataset[idx][1] for idx in indices])
            # preprocess for the cosine similarity computation
            for key, val in labels_cnt.items():
                clients_labels[client_id][key] += val

            client_path = os.path.join(PATH, mode, "task_{}".format(client_id))
            os.makedirs(client_path, exist_ok=True)

            train_indices, test_indices =\
                train_test_split(
                    indices,
                    train_size=args.tr_frac,
                    random_state=args.seed
                )

            if args.val_frac > 0:
                train_indices, val_indices = \
                    train_test_split(
                        train_indices,
                        train_size=1.-args.val_frac,
                        random_state=args.seed
                    )

                save_data(val_indices, os.path.join(client_path, "val.pkl"))

            save_data(train_indices, os.path.join(client_path, "train.pkl"))
            save_data(test_indices, os.path.join(client_path, "test.pkl"))
    
    clients_labels = np.asarray(clients_labels)
    similarity = rejust_cosine(clients_labels)
    np.save(os.path.join(PATH, 'similarity.npy'), similarity)


if __name__ == "__main__":
    main()