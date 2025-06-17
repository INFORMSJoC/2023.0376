# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from .wordlm_transformer import WordLMTransformer
from .emnist_convnet import EmnistConvNet
from .emnist_resnet import EmnistResNetGN
from .gldv2_resnet import GLDv2ResNetGN
from .fedpop_emnist_resnet import Canonical_EmnistResNetGN
from .fedpop_wordlm_transformer import Canonical_WordLMTransformer


def get_model_from_args(args, device):
    summary_args = dict(device=device)
    # print(f"the algorithm is {args.pfl_algo}")
    if args.dataset == 'emnist':
        if args.model_name in ['conv', 'convnet', None]:
            print('Running EMNIST with simple ConvNet')
            model = EmnistConvNet()
        elif args.model_name in ['resnet', 'resnet_gn']:
            print('Running EMNIST with ResNet18 w/ group norm')
            if args.personalize_on_client == "canonical" or ("pfl_algo" in args and args.pfl_algo.lower() == "fedpop"):
                model = Canonical_EmnistResNetGN(args.n_canonical, args.interpolate)
            else:
                model = EmnistResNetGN()
    elif args.dataset == 'stackoverflow':
        total_vocab_size = args.vocab_size + args.num_oov_buckets + 3  # add pad, bos, eos
        # TODO: adding fedpop
        if args.personalize_on_client == "canonical" or ("pfl_algo" in args and args.pfl_algo.lower() == "fedpop"):
            model = Canonical_WordLMTransformer(
                args.n_canonical, args.max_sequence_length, total_vocab_size, args.input_dim, args.attn_hidden_dim, args.fc_hidden_dim,
                args.num_attn_heads, args.num_transformer_layers, interpolate=args.interpolate,
                tied_weights=True, dropout_tr=args.dropout_tr, dropout_io=args.dropout_io,
            )
        else:
            model = WordLMTransformer(
                args.max_sequence_length, total_vocab_size, args.input_dim, args.attn_hidden_dim, args.fc_hidden_dim,
                args.num_attn_heads, args.num_transformer_layers, 
                tied_weights=True, dropout_tr=args.dropout_tr, dropout_io=args.dropout_io,
            )
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')

    return model.to(device)
    
