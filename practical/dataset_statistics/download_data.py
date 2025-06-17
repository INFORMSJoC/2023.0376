import argparse

import tensorflow_federated as tff

def main(args):
    print(f"downloading the stackoverflow dataset {args.cache_dir}")
    dataset = tff.simulation.datasets.stackoverflow.load_data(cache_dir=args.cache_dir)
    print(f"Finished!")
    print(f"downloading the emnist dataset to {args.cache_dir}")
    dataset = tff.simulation.datasets.emnist.load_data(only_digits=False, cache_dir=args.cache_dir)
    print(f"Finished!")

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--cache_dir',
        type=str,
        required=True)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    main(args)

