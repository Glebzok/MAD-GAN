from models import LSTMGenerator, LSTMDiscriminator, CNN_LSTMGenerator, CNN_LSTMDiscriminator, weights_init
import torch
import torch.nn as nn
torch.use_deterministic_algorithms(True)

from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle

from evaluation import evaluate_anomaly_detection
from anomaly_detection import covariance_similarity


def init(settings, device='cuda:0'):
    if settings['generator'].__name__ == 'LSTMGenerator':
        generator = LSTMGenerator(latent_space_dim=settings['latent_dim'],
                                  out_space_dim=settings['num_signals'],
                                  lstm_layers=settings['lstm_layers_g'],
                                  lstm_hidden_dim=settings['lstm_hidden_dim_g']).to(device)
    else:
        generator = CNN_LSTMGenerator(latent_space_dim=settings['latent_dim'],
                                      lstm_layers=settings['lstm_layers_g'],
                                      lstm_hidden_dim=settings['lstm_hidden_dim_g'],
                                      cnn_features=settings['cnn_features_g']).to(device)

    if settings['discriminator'].__name__ == 'LSTMDiscriminator':
        discriminator = LSTMDiscriminator(out_space_dim=settings['num_signals'],
                                          lstm_layers=settings['lstm_layers_d'],
                                          lstm_hidden_dim=settings['lstm_hidden_dim_d']).to(device)
    else:
        discriminator = CNN_LSTMDiscriminator(lstm_hidden_dim=settings['lstm_hidden_dim_d'],
                                              lstm_layers=settings['lstm_layers_d'],
                                              cnn_features=settings['cnn_features_d']).to(device)

    generator.apply(weights_init)
    discriminator.apply(weights_init)

    criterion = nn.BCELoss().to(device)

    optimizer_D = torch.optim.Adam(discriminator.parameters())
    optimizer_G = torch.optim.Adam(generator.parameters())

    dset = settings['data'](settings['normal_data_path'],
                            settings['abnormal_data_path'],
                            normal_label=settings['normal_label'],
                            abnormal_label=settings['abnormal_label'],
                            seq_length=settings['seq_length'],
                            seq_step=settings['seq_step'],
                            num_signals=settings['num_signals'])

    train_loader = torch.utils.data.DataLoader(dset.normal_data, batch_size=settings['batch_size'], shuffle=False)
    evaluate_loader = torch.utils.data.DataLoader(dset.all_data, batch_size=settings['batch_size'], shuffle=True)

    fake_label = settings['abnormal_label']

    return {'n_epochs': settings['num_epochs'], 'criterion': criterion, 'generator': generator,
            'discriminator': discriminator, 'optimizer_D': optimizer_D, 'optimizer_G': optimizer_G,
            'train_loader': train_loader, 'evaluate_loader': evaluate_loader, 'device': device,
            'fake_label': fake_label,
            'settings': settings}, dset


def train(n_epochs, criterion, generator, discriminator, optimizer_D, optimizer_G, train_loader, device,
          fake_label, settings, visualize_generation=False, evaluate_model=False, evaluate_loader=None, save_model=False, model_name=None):
    torch.manual_seed(0)

    latent_space_dim = settings['latent_dim']
    D_rounds = settings['D_rounds']
    G_rounds = settings['G_rounds']

    visualization_noise = torch.randn(1, settings['seq_length'], latent_space_dim, device=device)

    D_losses = []
    G_losses = []


    pr_scores, rec_scores, f1_scores = [], [], []

    for epoch in range(n_epochs):

        generator.train()
        discriminator.train()

        for i, data in tqdm(enumerate(train_loader), leave=False):

            # Real data
            real = data[0].to(torch.float).to(device)
            real_labels = data[1].to(torch.float).to(device)

            batch_size, seq_len = real.size(0), real.size(1)

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################

            for r in range(D_rounds):
                # Train with real data
                discriminator.zero_grad()

                output = discriminator(real)
                errD_real = criterion(output, real_labels)
                errD_real.backward()

                # Train with fake data

                # Fake data
                noise = torch.randn(batch_size, seq_len, latent_space_dim, device=device)
                fake = generator(noise)
                fake_labels = torch.full((batch_size, seq_len, 1), fake_label, device=device).to(torch.float)

                output = discriminator(fake.detach())
                errD_fake = criterion(output, fake_labels)
                errD_fake.backward()

                errD = errD_fake + errD_real
                optimizer_D.step()

            D_losses.append(errD.item())
            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################

            for r in range(G_rounds):
                generator.zero_grad()

                # Fake data
                noise = torch.randn(batch_size, seq_len, latent_space_dim, device=device)
                fake = generator(noise)

                output = discriminator(fake)
                errG = criterion(output, real_labels)
                errG.backward()
                optimizer_G.step()

            G_losses.append(errG.item())

        # Calc losses at epoch end:
        output = discriminator(real)
        errD_real = criterion(output, real_labels)
        D_x = output.mean().item()

        noise = torch.randn(batch_size, seq_len, latent_space_dim, device=device)
        fake = generator(noise).detach()

        output = discriminator(fake)
        errD_fake = criterion(output, fake_labels)
        D_G_z1 = output.mean().item()

        errD = (errD_real + errD_fake).item()
        errG = criterion(output, real_labels).item()

        # Report metrics
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f'
              % (epoch, n_epochs, i, len(train_loader),
                 errD, errG, D_x, D_G_z1), end='\n')

        if visualize_generation:
            # Visualize generation
            visualization_fake = generator(visualization_noise).detach().cpu()
            settings['plot_generated'](visualization_fake, settings, epoch)

        if evaluate_model:
            pr, rec, f1 = \
                evaluate_anomaly_detection(evaluate_loader, generator, discriminator, torch.optim.RMSprop, covariance_similarity,
                                           1e-3, 100, latent_space_dim,
                                           settings['lambda'], settings['tau'], fake_label, device)

            print('Precision: %.2f, Recall: %.2f, F1-score: %.2f'%(pr, rec, f1))

            pr_scores.append(pr)
            rec_scores.append(rec)
            f1_scores.append(f1)


        if save_model:
            # Save model
            if epoch % 10 == 0:
                with open('./models/model_{0}_{1}.pkl'.format(model_name, epoch), 'wb') as f:
                    pickle.dump([generator, discriminator, settings], f)

    if evaluate_model:
        metrics = pd.DataFrame(np.array([pr_scores, rec_scores, f1_scores]).T, columns=['precision', 'recall', 'f1'])
        print(metrics)
        metrics.to_csv('./experiments_results/metrics_evolve_{0}.csv'.format(model_name))

    return D_losses, G_losses
