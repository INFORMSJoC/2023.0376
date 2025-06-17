import os
import random
import argparse
import pickle
from matplotlib.pyplot import axes

import numpy as np

from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import ConcatDataset

from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity


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
        '--limit_data',
        help='the datasize is heterogeneous among clients',
        action='store_true'
    )
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

    return parser.parse_args()

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

def rejust_cosine(matrix) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    matrix -= np.mean(matrix, axis=0)
    res = cosine_similarity(matrix)
    return 0.5 + 0.5 * res

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

    # compute the number of clients per domain.
    per_domain_clients, clients_residual = n_clients // n_domain, n_clients % n_domain
    domain_clients = [per_domain_clients] * n_domain
    for i in range(clients_residual):
        domain_clients[i] += 1

    rng = random.Random(args.seed)
    client_sampleidx = []

    all_labels = np.array([data[1] for data in dataset])
    print(f'there are {len(all_labels)} samples')
    
    for i, domain in enumerate(domains):
        domain_idx = []
        # collect all labels' indexes in the specific domain
        for label in domain:
            label_idx = np.where(all_labels == label)[0]
            domain_idx.extend(label_idx)
        rng.shuffle(domain_idx)

        # TODO: add limited option
        domain_size = len(domain_idx)
        per_client_sample, sample_residual = divmod(domain_size, domain_clients[i])

        print(f'there are {domain_size} in {i}-th domain ({domain}), the average sample is {per_client_sample} and the number of clients is {domain_clients[i]}')

        start_idx = 0
        for _ in range(sample_residual):
            t = domain_idx[start_idx:start_idx+per_client_sample+1]
            client_sampleidx.append(t)
            start_idx += per_client_sample+1
        
        for _ in range(sample_residual, domain_clients[i]):
            t = domain_idx[start_idx:start_idx+per_client_sample]
            client_sampleidx.append(t)
            start_idx += per_client_sample
    
    client_idx = list(range(n_clients))
    rng.shuffle(client_idx)

    index_map = OrderedDict()
    for i, idx in enumerate(client_idx):
        index_map[idx] = i
    with open('client_index.pkl', "wb") as f:
        pickle.dump([index_map[i] for i in range(n_clients)], f)

    client_sampleidx = [client_sampleidx[i] for i in client_idx]

    print(f'There are total {len(client_sampleidx)} clients')
    print(f'The number of samples in each client is')
    print([len(element) for element in client_sampleidx])
    if args.test_tasks_frac > 0:
        raise 'unsupported error!'
        # train_clients_indices, test_clients_indices = \
        #     train_test_split(clients_indices, test_size=args.test_tasks_frac, random_state=args.seed)
    else:
        # no new clients.
        train_clients_indices, test_clients_indices = client_sampleidx, []

    os.makedirs(os.path.join(PATH, "train"), exist_ok=True)
    os.makedirs(os.path.join(PATH, "test"), exist_ok=True)

    clients_labels = [[0] * N_CLASSES for _ in range(n_clients)]

    for mode, clients_indices in [('train', train_clients_indices), ('test', test_clients_indices)]:
        for client_id, indices in enumerate(clients_indices):
            if len(indices) == 0:
                continue

            labels_cnt = Counter([dataset[idx][1] for idx in indices])
            client_labels = [0] * N_CLASSES
            # preprocess for the cosine similarity computation
            for key, val in labels_cnt.items():
                client_labels[key] += val
            
            for label in range(N_CLASSES):
                clients_labels[client_id][label] += client_labels[label]

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
    similarity = cosine_similarity(clients_labels)
    np.save(os.path.join(PATH, 'similarity.npy'), similarity)


if __name__ == "__main__":
    main()