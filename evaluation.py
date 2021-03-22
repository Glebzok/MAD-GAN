from tqdm import tqdm
import numpy as np
import torch
torch.use_deterministic_algorithms(True)
from anomaly_detection import detect_anomalies

from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score

def evaluate_anomaly_detection(data_loader, generator, discriminator,
                               reconstruction_optimizer, signals_similarity_measure,
                               reconstruction_tol, reconstruction_max_iter, latent_space_dim,
                               lmbda, tau, fake_label, device):
    torch.manual_seed(0)

    if tau is None:
        predicted_scores = []
        labels = []

        for batch, batch_label in tqdm(data_loader):
            predicted_score, _ = \
                detect_anomalies(batch, generator, discriminator,
                                 reconstruction_optimizer, signals_similarity_measure,
                                 device, fake_label, latent_space_dim, lmbda, 0.5, reconstruction_tol,
                                 reconstruction_max_iter)  # don't care about tau, as we take only predictions

            predicted_scores.append(predicted_score.flatten())
            labels.append(batch_label[:, :, 0].cpu().numpy().astype('int').flatten())

        predicted_scores = np.concatenate(predicted_scores).flatten()
        labels = np.concatenate(labels).flatten()

        precision_vals, recall_vals, taus = precision_recall_curve(labels, predicted_scores)

        f1 = 2 * precision_vals * recall_vals / (precision_vals + recall_vals)

        return precision_vals, recall_vals, f1, taus

    else:
        predicted_labels = []
        labels = []

        for batch, batch_label in tqdm(data_loader):
            _, predicted_label = \
                detect_anomalies(batch, generator, discriminator,
                                 reconstruction_optimizer, signals_similarity_measure,
                                 device, fake_label, latent_space_dim, lmbda, tau, reconstruction_tol,
                                 reconstruction_max_iter)

            predicted_labels.append(predicted_label.flatten())
            labels.append(batch_label[:, :, 0].cpu().numpy().astype('int').flatten())

        predicted_labels = np.concatenate(predicted_labels).flatten()
        labels = np.concatenate(labels).flatten()
        precision = precision_score(labels, predicted_labels)
        recall = recall_score(labels, predicted_labels)
        f1 = f1_score(labels, predicted_labels)

        return precision, recall, f1
