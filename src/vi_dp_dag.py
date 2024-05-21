import argparse
from src.run_probabilistic_dag_autoencoder import run

param_dict = {
    'seed_dataset': 123,  # Seed to shuffle dataset. int
    'dataset_name': 'all',  # Dataset name. string
    'dataset_directory': 'to_complete',  # Dataset directory. string
    'i_dataset': 1,  # Dataset name. string
    'split': [.8, .1, .1],  # Split for train/val/test sets. list of floats

    # Architecture parameters
    'seed_model': 123,  # Seed to init model. int
    'directory_model': 'to_complete',  # Path to save model. string
    'ma_hidden_dims': [16, 16, 16],  # Output dimension. int
    'ma_architecture': 'linear',  # Output dimension. int
    'ma_fast': False,  # Output dimension. int
    'pd_initial_adj': 'Learned',  # Output dimension. int
    'pd_temperature': 1.0,  # Output dimension. int
    'pd_hard': True,  # Output dimension. int
    'pd_order_type': 'topk',  # Output dimension. int
    'pd_noise_factor': 1.0,  # Hidden dimensions. list of ints

    # Training parameters
    'directory_results': 'to_complete',  # Path to save resutls. string
    'max_epochs': 100,  # Maximum number of epochs for training
    'patience': 10,  # Patience for early stopping. int
    'frequency': 2,  # Frequency for early stopping test. int
    'batch_size': 64,  # Batch size. int
    'ma_lr': 1e-3,  # Learning rate. float
    'pd_lr': 1e-2,  # Learning rate. float
    'loss': 'ELBO',  # Loss name. string
    'regr': .01,  # Regularization factor in Bayesian loss. float
    'prior_p': .01  # Regularization factor in Bayesian loss. float
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data-dir", type=str, default="./", help="base results directory"
)

args = parser.parse_args()
param_dict['dataset_directory'] = args.data_dir
param_dict['directory_model'] = 'model_dds/' + args.data_dir
param_dict['directory_results'] = 'results_dds' + args.data_dir

results = run(**param_dict)

