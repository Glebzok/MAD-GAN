from settings import settings
from train import init, train
import pickle
import numpy as np
import pandas as pd
from evaluation import evaluate_anomaly_detection
from anomaly_detection import covariance_similarity
import torch

def train_models(device):
    for dataset_name in ['kdd99', 'WADI', 'SWaT']:
        train_params, dset = init(settings[dataset_name], device)
        train(**train_params, visualize_generation=True, evaluate_model=True, save_model=False, model_name=dataset_name)

        with open('./experiments_results/model_' + dataset_name + '.pkl', 'wb') as f:
            pickle.dump([train_params['generator'], train_params['discriminator'], train_params['settings']], f)


def evaluate_models(device):
    n_lambdas = 10
    lambdas = np.linspace(0, 1, n_lambdas)

    for dataset_name in ['kdd99', 'WADI', 'SWaT']:
        with open('./experiments_results/model_' + dataset_name + '.pkl', 'rb') as f:
            generator, discriminator, train_params = pickle.load(f)

        dset = train_params['data'](train_params['normal_data_path'],
                                    train_params['abnormal_data_path'],
                                    normal_label=train_params['normal_label'],
                                    abnormal_label=train_params['abnormal_label'],
                                    seq_length=train_params['seq_length'],
                                    seq_step=train_params['seq_step'],
                                    num_signals=train_params['num_signals'])
        evaluate_loader = torch.utils.data.DataLoader(dset.all_data, batch_size=train_params['batch_size'], shuffle=True)

        metric_values = {}
        for lambd in lambdas:
            precision_vals, recall_vals, f1, thresholds = \
                evaluate_anomaly_detection(
                evaluate_loader,
                generator, discriminator,
                torch.optim.RMSprop, covariance_similarity, 1e-3, 100, train_params['latent_dim'],
                lambd, None, train_params['normal_label'], device)

            best_indices = [np.argmax(precision_vals[precision_vals < 1]),
                            np.argmax(recall_vals[recall_vals < 1]),
                            np.argmax(f1)]

            thresholds = np.hstack((thresholds, -1))
            metrics = np.vstack([precision_vals, recall_vals, f1, thresholds])

            metric_values[lambd] = [metrics[:, best_indices[0]],
                                    metrics[:, best_indices[1]],
                                    metrics[:, best_indices[2]]]

        print(metric_values)

        with open('./experiments_results/metrics_' + dataset_name + '.pkl', 'wb') as f:
            pickle.dump(metric_values, f)

    for dataset_name in ['kdd99', 'WADI', 'SWaT']:
        with open('./experiments_results/metrics_' + dataset_name + '.pkl', 'rb') as f:
            metrics = pickle.load(f)

        for i in range(3):
            lst = []
            for lambd in metrics.keys():
                lst.append(metrics[lambd][i][i])
            best_index = np.nanargmax(lst)
            best_lambda = lambdas[best_index]
            best_tau = metrics[best_lambda][i][3]
            best_metric = metrics[best_lambda][i][:-1]

            print('{}: best lambda = {}, best tau = {}, \nprecision = {}, recall = {}, f-score = {}\n'.format(
                dataset_name,
                best_lambda,
                best_tau,
                best_metric[0],
                best_metric[1],
                best_metric[2]))
        print('\n')


def pca_components(device):
    metric_values = []
    components_grid = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for n_components in components_grid:
        params = settings['SWaT'].copy()
        params['num_signals'] = n_components
        train_params, dset = init(params, device)
        train(**train_params, visualize_generation=False, evaluate_model=False, save_model=False)

        dset = params['data'](params['normal_data_path'],
                            params['abnormal_data_path'],
                            normal_label=params['normal_label'],
                            abnormal_label=params['abnormal_label'],
                            seq_length=params['seq_length'],
                            seq_step=params['seq_step'],
                            num_signals=params['num_signals'])
        evaluate_loader = torch.utils.data.DataLoader(dset.all_data, batch_size=params['batch_size'],
                                                      shuffle=True)

        metric_values.append(evaluate_anomaly_detection(
            evaluate_loader,
            train_params['generator'], train_params['discriminator'],
            torch.optim.RMSprop, covariance_similarity, 1e-3, 100, params['latent_dim'],
            params['lambda'], params['tau'], params['normal_label'], device))

    metric_values = pd.DataFrame(metric_values, columns=['precision', 'recall', 'f1'], index=components_grid)
    print("Metrics dependency on number of PCA components:\n", metric_values)
    metric_values.to_csv('./experiments_results/pca_metrics.csv')


def seq_length(device):
    for dataset_name in ['WADI', 'SWaT']:
        metric_values = []
        seq_length_grid = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300]
        for seq_length in seq_length_grid:
            params = settings['SWaT'].copy()
            params['seq_length'] = seq_length
            train_params, dset = init(params, device)
            train(**train_params, visualize_generation=False, evaluate_model=False, save_model=False)

            dset = params['data'](params['normal_data_path'],
                                params['abnormal_data_path'],
                                normal_label=params['normal_label'],
                                abnormal_label=params['abnormal_label'],
                                seq_length=params['seq_length'],
                                seq_step=params['seq_step'],
                                num_signals=params['num_signals'])
            evaluate_loader = torch.utils.data.DataLoader(dset.all_data, batch_size=params['batch_size'],
                                                          shuffle=True)

            metric_values.append(evaluate_anomaly_detection(
                evaluate_loader,
                train_params['generator'], train_params['discriminator'],
                torch.optim.RMSprop, covariance_similarity, 1e-3, 100, params['latent_dim'],
                params['lambda'], params['tau'], params['normal_label'], device))

        metric_values = pd.DataFrame(metric_values, columns=['precision', 'recall', 'f1'], index=seq_length_grid)
        print("Metrics dependency on sequence length ({}):\n".format(dataset_name), metric_values)
        metric_values.to_csv('./experiments_results/seq_length_metrics_{0}.csv'.format(dataset_name))


def run_all(device):

    train_models(device)
    evaluate_models(device)
    pca_components(device)
    seq_length(device)
