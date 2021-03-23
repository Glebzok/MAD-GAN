import pickle
import numpy as np
import os
from tqdm import tqdm

import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms

from mnist_encoder import idx2onehot, IM_SIZE, CVAE, Encoder, Decoder


def process_pair_xy(x, y):
    x = x.view(-1, 28 * 28)
    x = x.to(device)
    # img = x.view(28, 28).data

    # convert y into one-hot encoding
    y = idx2onehot(y.view(-1, 1))
    y = y.to(device)
    return x, y


def reparameterize_and_sample(z_mu, z_var):
    std = torch.exp(z_var / 2)
    eps = torch.randn_like(std)
    x_sample = eps.mul(std).add_(z_mu)

    return x_sample


def apply_model(model, x, y):
    _, z_mu, z_var = model(x, y)
    x_sample = reparameterize_and_sample(z_mu, z_var)
    return x_sample


def seq_from_pair_imgs(model, first_x, first_y, second_x, second_y, seq_len):
    first_x_sample = apply_model(model, first_x, first_y)
    second_x_sample = apply_model(model, second_x, second_y)

    image_weights = np.linspace(0, 1, seq_len)
    image_sequence = []
    label_sequence = []

    for weight in image_weights:
        intermediate_x = (1 - weight) * first_x_sample + weight * second_x_sample
        intermediate_label = (1 - weight) * first_y + weight * second_y
        z = torch.cat((intermediate_x, intermediate_label), dim=1)

        # decode
        generated_intermediate_x = model.decoder(z)
        image_sequence.append(generated_intermediate_x)
        label_sequence.append(intermediate_label[0].argmax().item())

    return image_sequence, label_sequence


def save_img_series(path_to_folder, iter_number, img_sequence, label_sequence, rewrite=True):
    path_to_directory = os.path.join(path_to_folder, 'img_series_{}'.format(iter_number))
    if not rewrite:
        os.mkdir(path_to_directory)
    else:
        if not os.path.exists(path_to_directory):
            os.mkdir(path_to_directory)

    for i in range(0, len(img_sequence)):
        path_to_file = os.path.join(path_to_directory, '{}_{}.pt'.format(i, label_sequence[i]))
        torch.save(img_sequence[i].view(IM_SIZE, IM_SIZE), path_to_file)


def generate_normal_sequence(normal_number, sequence_length, num_it, model, dataset, path_to_folder):
    """
    normal_number - sequence of this number will be generated,
    sequence_length - length of image sequence,
    num_it - number of sequences to generate,
    model - sequence generator,
    dataset - mnist dataset,
    path_to_folder - save sequences path
    """
    idx = torch.tensor(test_dataset.targets) == normal_number

    number_dataset = torch.utils.data.dataset.Subset(dataset, np.where(idx == 1)[0])

    first_number_dataloader = DataLoader(number_dataset, batch_size=1, shuffle=True)
    second_number_dataloader = DataLoader(number_dataset, batch_size=1, shuffle=True)

    iteration_number = num_it
    seq_len = sequence_length
    for i, (x, y) in zip(range(iteration_number), first_number_dataloader):
        # from 1 to 1
        first_x, first_y = process_pair_xy(x, y)
        second_x, second_y = next(iter(second_number_dataloader))
        second_x, second_y = process_pair_xy(second_x, second_y)

        image_sequence, labels_sequence = seq_from_pair_imgs(model, first_x, first_y,
                                                             second_x, second_y, seq_len)

        save_img_series(path_to_folder, i, image_sequence, labels_sequence)
    return image_sequence, labels_sequence


def generate_sequence_with_anomalies(normal_number, anomaly_number, sequence_length, ratio, num_it, model, dataset,
                                     path_to_folder):
    """
    normal_number - sequence of this number will be generated,
    anomaly_number - anomaly number in sequence,
    sequence_length - length of image sequence,
    num_it - number of sequences to generate,
    ratio - ratio of anomalies in sequence
    model - sequence generator,
    dataset - mnist dataset,
    path_to_folder - save sequences path
    """

    number_of_anomaly_elements = int(ratio * sequence_length)

    idx_1 = torch.tensor(test_dataset.targets) == normal_number
    idx_2 = torch.tensor(test_dataset.targets) == anomaly_number

    normal_dataset = torch.utils.data.dataset.Subset(test_dataset, np.where(idx_1 == 1)[0])
    anomaly_dataset = torch.utils.data.dataset.Subset(test_dataset, np.where(idx_2 == 1)[0])

    first_normal_dataloader = DataLoader(normal_dataset, batch_size=1, shuffle=True)
    second_normal_dataloader = DataLoader(normal_dataset, batch_size=1, shuffle=True)
    third_normal_dataloader = DataLoader(normal_dataset, batch_size=1, shuffle=True)

    first_anomaly_dataloader = DataLoader(anomaly_dataset, batch_size=1, shuffle=True)

    iteration_number = num_it
    seq_len = sequence_length

    # w = np.random.randint(1, 10)
    anomaly_start_idx = np.random.randint(number_of_anomaly_elements, sequence_length - number_of_anomaly_elements)

    for i, (x, y) in zip(range(iteration_number), first_normal_dataloader):
        # from normal_number to normal_number
        first_x, first_y = process_pair_xy(x, y)
        second_x, second_y = next(iter(second_normal_dataloader))
        second_x, second_y = process_pair_xy(second_x, second_y)

        # first_seq = np.random.randint(1, seq_len//2)
        first_seq = anomaly_start_idx - number_of_anomaly_elements
        image_sequence_first_part, labels_first_part = seq_from_pair_imgs(model, first_x, first_y,
                                                                          second_x, second_y,
                                                                          first_seq)

        # from normal_number to anomaly_number
        third_x, third_y = next(iter(first_anomaly_dataloader))
        third_x, third_y = process_pair_xy(third_x, third_y)

        # second_seq = np.random.randint(first_seq, first_seq + w)
        second_seq = number_of_anomaly_elements
        image_sequence_second_part, labels_second_part = seq_from_pair_imgs(model, second_x, second_y,
                                                                            third_x, third_y, second_seq)

        # from anomaly_number to normal_number
        fourth_x, fourth_y = next(iter(second_normal_dataloader))
        fourth_x, fourth_y = process_pair_xy(fourth_x, fourth_y)

        # second_seq = np.random.randint(first_seq, first_seq + w)
        third_seq = number_of_anomaly_elements
        image_sequence_third_part, labels_third_part = seq_from_pair_imgs(model, third_x, third_y,
                                                                          fourth_x, fourth_y, third_seq)

        # from normal_number to normal_number

        fifth_x, fifth_y = next(iter(third_normal_dataloader))
        fifth_x, fifth_y = process_pair_xy(fifth_x, fifth_y)

        image_sequence_fourth_part, labels_fourth_part = seq_from_pair_imgs(model, fourth_x, fourth_y,
                                                                            fifth_x, fifth_y,
                                                                            seq_len - third_seq - second_seq - first_seq)

        full_image_sequence = image_sequence_first_part + image_sequence_second_part + image_sequence_third_part + image_sequence_fourth_part
        full_labels_sequence = labels_first_part + labels_second_part + labels_third_part + labels_fourth_part

        save_img_series(path_to_folder, i, full_image_sequence, full_labels_sequence)
    return full_image_sequence, full_labels_sequence


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open("./models/model_seq_gen.pkl", "rb") as input_file:
        model = pickle.load(input_file)

    model.to(device)

    test_dataset = MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]))

    if not os.path.exists('./project_norm_seq'):
        os.mkdir('./project_norm_seq')
    if not os.path.exists('./project_seq_test'):
        os.mkdir('./project_seq_test')

    path_to_anomaly_seq_folder = './project_seq_test'
    path_to_normal_folder = './project_norm_seq'

    train_data_size = 1000
    test_data_size = 1000

    seq_len = 1

    train_seqs = []
    train_labels = []

    for _ in tqdm(range(train_data_size)):
        num = np.random.randint(0, 10)
        image_sequence, _ = generate_normal_sequence(num, seq_len, 1, model, test_dataset, path_to_normal_folder)
        seq_labels = np.zeros(seq_len)

        train_seqs.append(image_sequence)
        train_labels.append(seq_labels)

    train_seqs = torch.stack([torch.cat(i) for i in train_seqs]).reshape((train_data_size, seq_len, 28, 28))
    train_labels = np.stack(train_labels)

    with open('./data/MNIST_train_seq.pkl', 'wb') as f:
        pickle.dump([train_seqs.detach().cpu().numpy(), train_labels], f)

    test_seqs = []
    test_labels = []

    for _ in tqdm(range(test_data_size)):
        num1, num2 = np.random.choice(10, 2, replace=False)

        num2_frac = np.random.uniform(max(0.1, 3.5 / test_data_size), 0.5)

        image_sequence, seq_labels = generate_sequence_with_anomalies(num1, num2, seq_len, num2_frac, 1, model,
                                                                      test_dataset, path_to_anomaly_seq_folder)
        seq_labels = np.array(np.array(seq_labels) != num1).astype(int)

        test_seqs.append(image_sequence)
        test_labels.append(seq_labels)

    test_seqs = torch.stack([torch.cat(i) for i in test_seqs]).reshape((test_data_size, seq_len, 28, 28))
    test_labels = np.stack(test_labels)

    with open('./data/MNIST_test_seq.pkl', 'wb') as f:
        pickle.dump([test_seqs.detach().cpu().numpy(), test_labels], f)
