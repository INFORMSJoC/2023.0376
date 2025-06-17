from collections import OrderedDict
import pickle
import time
import random
import argparse

import numpy as np
import pandas as pd
from scipy import sparse

import torch
from torch import nn
import tensorflow as tf
import tensorflow_federated as tff

from tqdm import tqdm


vocab_size = 10000 
max_sequence_length = 20
max_num_elements_per_client = 1000
every_log_sample = 1000

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--cache_dir',
        type=str,
        required=True)
    
    return parser.parse_args()

def main(args):
    train_ids = pd.read_csv("stackoverflow_client_ids_train.csv",dtype=str)
    test_ids = pd.read_csv("stackoverflow_client_ids_test.csv", dtype=str)
    train_sizes = pd.read_csv("stackoverflow_client_sizes_train.csv", header=0, names=["id", "num"], dtype={"id":str, "num":int})
    test_sizes = pd.read_csv("stackoverflow_client_sizes_test.csv", header=0, names=["id", "num"], dtype={"id":str, "num":int})
    train_ids = train_ids["train_ids"].to_list()
    test_ids = test_ids["test_ids"].to_list()

    train_sizes.set_index("id", inplace=True)
    test_sizes.set_index("id", inplace=True)

    vocab_dict = tff.simulation.datasets.stackoverflow.load_word_counts(cache_dir=args.cache_dir)
    vocab = list(vocab_dict.keys())[:vocab_size]

    table_values = np.arange(len(vocab), dtype=np.int64)
    table = tf.lookup.StaticVocabularyTable(
        tf.lookup.KeyValueTensorInitializer(vocab, table_values),
        num_oov_buckets = 1
    )

    def tokenize_fn(example):
        sentence = tf.reshape(example['tokens'], shape=[1])
        words = tf.strings.split(sentence, ' ').values
        truncated_words = words[:max_sequence_length]
        token = table.lookup(truncated_words)
        return token
    
    def sparse_count(sample):
        unique_words, _,  count = tf.unique_with_counts(sample, out_idx=tf.dtypes.int64)
        res = tf.SparseTensor(indices=tf.expand_dims(unique_words, -1), values=count, dense_shape=[vocab_size+1])
        res = tf.sparse.reorder(res)
        return res

    def dataset_sparse_add(x, y):
        return tf.sparse.reorder(tf.sparse.add(x, y))


    dataset = tff.simulation.datasets.stackoverflow.load_data(cache_dir=args.cache_dir)
    train_dataset, test_dataset = dataset[0], dataset[2]

    client_label_d = OrderedDict()
    sample_cnt = 0

    for id in tqdm(train_ids, desc="Process train label distritbuion"):
        dataset_size = train_sizes.loc[id].num
        dataset_size = min(dataset_size, max_num_elements_per_client)
        
        sparse_dataset = train_dataset.create_tf_dataset_for_client(id).take(dataset_size).map(tokenize_fn)
        res = sparse_dataset.map(sparse_count).reduce(tf.SparseTensor(indices = tf.constant([[vocab_size]],dtype=tf.int64), values = tf.constant([0], dtype=tf.int64), dense_shape =[vocab_size+1]), dataset_sparse_add)
        value = res.values.numpy()
        indices = tf.squeeze(res.indices).numpy()
        coo_matrix = sparse.coo_matrix((value, (np.zeros(len(value)), indices)), shape=(1, vocab_size+1))
        client_label_d[id] = coo_matrix
        sample_cnt += 1
        if sample_cnt % every_log_sample == 0:
            print(f"Processing {sample_cnt}/117858")

    for id in tqdm(test_ids, desc="Process test label distritbuion"):
        dataset_size = train_sizes.loc[id].num
        dataset_size = min(dataset_size, max_num_elements_per_client)

        sparse_dataset = test_dataset.create_tf_dataset_for_client(id).map(tokenize_fn)
        res = sparse_dataset.map(sparse_count).reduce(tf.SparseTensor(indices = tf.constant([[vocab_size]],dtype=tf.int64), values = tf.constant([0], dtype=tf.int64), dense_shape =[vocab_size+1]), dataset_sparse_add)
        value = res.values.numpy()
        indices = tf.squeeze(res.indices).numpy()
        coo_matrix = sparse.coo_matrix((value, (np.zeros(len(value)), indices)), shape=(1, vocab_size+1))
        if id in client_label_d:
            client_label_d[id] += coo_matrix
        else:
            client_label_d[id] = coo_matrix
        sample_cnt += 1
        if sample_cnt % every_log_sample == 0:
            print(f"Processing {sample_cnt}/117858")


    concate_matrix = sparse.vstack([client_label_d[key] for key in client_label_d])
    sparse.save_npz('concate_matrix.npz', concate_matrix)

    concate_matrix_csr = sparse.csr_matrix(concate_matrix)
    without_oov = concate_matrix_csr[:, :vocab_size]

    save_path = "stackoverflow.npz"
    sparse.save_npz(save_path, without_oov)
    print(f"The vocabulary sparse matrix is stored in {save_path}")

    return 


if __name__ == "__main__":
    args = parse_args()
    main(args)