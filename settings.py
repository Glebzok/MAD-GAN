from data import KddDataset, WadiDataset, SwatDataset, MnistDataset
from models import LSTMGenerator, LSTMDiscriminator, CNN_LSTMGenerator, CNN_LSTMDiscriminator
from generation_results_plotting import plot_1d_generated, plot_2d_generated

settings = {'kdd99': {
    "data": KddDataset,
    "normal_data_path": "./data/kdd99_train.npy",
    "abnormal_data_path": "./data/kdd99_test.npy",

    "seq_length": 30,
    "num_signals": 6,
    "seq_step": 10,

    "normal_label": 0,
    "abnormal_label": 1,

    "generator": LSTMGenerator,
    "discriminator": LSTMDiscriminator,

    "latent_dim": 15,
    "lstm_hidden_dim_g": 100,
    "lstm_layers_g": 3,
    "lstm_hidden_dim_d": 100,
    "lstm_layers_d": 1,

    "batch_size": 500,
    "num_epochs": 100,
    "D_rounds": 1,
    "G_rounds": 3,

    "lambda": 0.1,
    "tau": 0.65,

    "plot_generated": plot_1d_generated
},
    'WADI': {
        "data": WadiDataset,
        "normal_data_path": "./data/WADI_14days.csv",
        "abnormal_data_path": "./data/WADI_attackdata.csv",

        "seq_length": 30,
        "num_signals": 8,
        "seq_step": 10,

        "normal_label": 0,
        "abnormal_label": 1,

        "generator": LSTMGenerator,
        "discriminator": LSTMDiscriminator,

        "latent_dim": 15,
        "lstm_hidden_dim_g": 100,
        "lstm_layers_g": 3,
        "lstm_hidden_dim_d": 100,
        "lstm_layers_d": 1,

        "batch_size": 500,
        "num_epochs": 100,
        "D_rounds": 1,
        "G_rounds": 3,

        "lambda": 0.1,
        "tau": 0.65,

        "plot_generated": plot_1d_generated
    },
    'SWaT': {
        "data": SwatDataset,
        "normal_data_path": "./data/SWaT_Dataset_Normal_v0.csv",
        "abnormal_data_path": "./data/SWaT_Dataset_Attack_v0.csv",

        "seq_length": 30,
        "num_signals": 5,
        "seq_step": 10,

        "normal_label": 0,
        "abnormal_label": 1,

        "generator": LSTMGenerator,
        "discriminator": LSTMDiscriminator,

        "latent_dim": 15,
        "lstm_hidden_dim_g": 100,
        "lstm_layers_g": 3,
        "lstm_hidden_dim_d": 100,
        "lstm_layers_d": 1,

        "batch_size": 500,
        "num_epochs": 100,
        "D_rounds": 1,
        "G_rounds": 3,

        "lambda": 0.1,
        "tau": 0.49,

        "plot_generated": plot_1d_generated
    },
    "MNIST": {
        "data": MnistDataset,
        "normal_data_path": './data/MNIST_3_train.pkl',
        "abnormal_data_path": './data/MNIST_3_test.pkl',

        "seq_length": 3,
        "num_signals": None,
        "seq_step": None,

        "normal_label": 0,
        "abnormal_label": 1,

        "generator": CNN_LSTMGenerator,
        "discriminator": CNN_LSTMDiscriminator,

        "latent_dim": 100,
        "lstm_hidden_dim_g": 100,
        "lstm_layers_g": 4,
        "lstm_hidden_dim_d": 100,
        "lstm_layers_d": 1,

        "cnn_features_g": 64,
        "cnn_features_d": 64,

        "batch_size": 128,
        "num_epochs": 500,
        "D_rounds": 2,
        "G_rounds": 10,

        "lambda": 0.5,
        "tau": 0.5,

        "plot_generated": plot_2d_generated
    }
}