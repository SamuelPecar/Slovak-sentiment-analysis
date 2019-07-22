import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Using {} device".format(device))

batch_size = 32

data_params = {
    'data_file': 'data/sentiment_extended.csv',
    'x_column': 1,
    'y_column': 2,
    'header': 0,
    'sep': '\t'
}

# data_params = {
#     'data_file': 'data/data.csv',
#     # 'data_file': 'data/processed_data.csv',
#     'x_column': 4,
#     'y_column': 2,
#     'header': 0,
#     'sep': ';'
# }

encoder_params = {
    'hidden_size': 512,
    'num_layers': 1,
    'bidirectional': True,
    # 'bidirectional': False,
    'dropout': 0.3,
    'batch_size': batch_size
}

model_params = {
    'output_dim': 3,
    'dropout': 0.5,
    'attention': True
}

embed_params = {
    'embedding_dropout': 0.3,
    'type': 'word2vec'
}

training_params = {
    'n_epochs': 20
}

paths = {
    'f1_score': {
        'model_path': 'checkpoints/best_valid_f1_model',
        'submission': 'submissions/f1_score'
    }
}

ensemble_models = [
    'run_1',
    # 'run_2',
    # 'run_3',
    'run_4',
    'run_5',
]