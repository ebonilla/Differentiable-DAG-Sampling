from src.probabilistic_dag_model.probabilistic_dag_autoencoder import ProbabilisticDAGAutoencoder
from src.probabilistic_dag_model.train_probabilistic_dag_autoencoder import train_autoencoder
from src.datasets.DAGDataset import get_dag_dataset
from src.run_probabilistic_dag_autoencoder import run

param_dict = {
    'seed_dataset': 123,  # Seed to shuffle dataset. int
    'dataset_name': 'data_p10_e10_n1000_GP',  # Dataset name. string
    'dataset_directory': 'src/datasets/datasets_dp_dag',  # Dataset directory. string
    'i_dataset': 1,  # Dataset name. string
    'split': [.8, .1, .1],  # Split for train/val/test sets. list of floats

    # Architecture parameters
    'seed_model': 123,  # Seed to init model. int
    'directory_model': 'models_dp_dag',  # Path to save model. string
    'ma_hidden_dims': [16, 16, 16],  # Output dimension. int
    'ma_architecture': 'linear',  # Output dimension. int
    'ma_fast': False,  # Output dimension. int
    'pd_initial_adj': 'Learned',  # Output dimension. int
    'pd_temperature': 1.0,  # Output dimension. int
    'pd_hard': True,  # Output dimension. int
    'pd_order_type': 'topk',  # Output dimension. int
    'pd_noise_factor': 1.0,  # Hidden dimensions. list of ints

    # Training parameters
    'directory_results': 'results_dp_dag',  # Path to save resutls. string
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

results = run(**param_dict)

