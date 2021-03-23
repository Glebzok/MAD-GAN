import torch
import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from torchvision.transforms import Resize


class SubData(torch.utils.data.Dataset):
    def __init__(self, samples, labels):
        super(SubData, self).__init__()

        self.samples = samples
        self.labels = labels

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


class Data(torch.utils.data.Dataset):
    def __init__(self, normal_data_path, abnormal_data_path, normal_label, abnormal_label, seq_length, seq_step,
                 num_signals):
        super(Data, self).__init__()

        self.scaler1 = None
        self.scaler2 = None
        self.pca = None

        normal_samples, normal_labels = self.preprocess_data(self.load_normal_data(normal_data_path),
                                                             True, normal_label, abnormal_label, seq_step, seq_length,
                                                             num_signals)
        self.normal_data = SubData(normal_samples, normal_labels)

        abnormal_samples, abnormal_labels = self.preprocess_data(self.load_abnormal_data(abnormal_data_path),
                                                                 False, normal_label, abnormal_label, seq_step,
                                                                 seq_length, num_signals)
        self.all_data = SubData(np.concatenate([normal_samples, abnormal_samples]),
                                np.concatenate([normal_labels, abnormal_labels]))

    def load_normal_data(self, data_path):
        raise NotImplementedError()

    def load_abnormal_data(self, data_path):
        raise NotImplementedError()

    def preprocess_data(self, samples, is_normal, normal_label, abnormal_label, seq_step, seq_length, num_signals):

        if is_normal:
            self.scaler1 = StandardScaler().fit(samples)

        samples = self.scaler1.transform(samples)

        label = normal_label if is_normal else abnormal_label
        labels = np.ones(samples.shape[0]) * label

        if is_normal:
            self.pca = PCA(num_signals, svd_solver='full').fit(samples)

        samples = self.pca.transform(samples)

        if is_normal:
            self.scaler2 = StandardScaler().fit(samples)

        samples = self.scaler2.transform(samples)

        num_samples = (samples.shape[0] - seq_length) // seq_step

        aa = np.empty([num_samples, seq_length, num_signals], dtype='float')
        bb = np.empty([num_samples, seq_length, 1], dtype='float')

        for j in range(num_samples):
            bb[j, :, :] = np.reshape(labels[(j * seq_step):(j * seq_step + seq_length)], [-1, 1])
            aa[j, :, :] = samples[(j * seq_step):(j * seq_step + seq_length), :]

        return aa.astype('float32'), bb.astype('float32')


class KddDataset(Data):
    def __init__(self, normal_data_path, abnormal_data_path, normal_label, abnormal_label, seq_length, seq_step,
                 num_signals):
        super(KddDataset, self).__init__(normal_data_path, abnormal_data_path, normal_label, abnormal_label, seq_length,
                                         seq_step, num_signals)

    def load_normal_data(self, data_path):
        return np.load(data_path)[:, :-1]

    def load_abnormal_data(self, data_path):
        return np.load(data_path)[:, :-1]


class SwatDataset(Data):
    def __init__(self, normal_data_path, abnormal_data_path, normal_label, abnormal_label, seq_length, seq_step,
                 num_signals):
        super(SwatDataset, self).__init__(normal_data_path, abnormal_data_path, normal_label, abnormal_label,
                                          seq_length, seq_step, num_signals)

    def load_normal_data(self, data_path):
        return pd.read_csv(data_path).drop([' Timestamp', 'Normal/Attack'], axis=1).dropna(axis=1).values[
               21600:]

    def load_abnormal_data(self, data_path):
        return pd.read_csv(data_path).drop([' Timestamp', 'Normal/Attack'], axis=1).dropna(axis=1).values[
               21600:]


class WadiDataset(Data):
    def __init__(self, normal_data_path, abnormal_data_path, normal_label, abnormal_label, seq_length, seq_step,
                 num_signals):
        self.drop_columns = ['Row', 'Date', 'Time',
                             r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_LS_001_AL',
                             r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_LS_002_AL',
                             r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_P_001_STATUS',
                             r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_P_002_STATUS']
        # metadata on contains NaNs

        super(WadiDataset, self).__init__(normal_data_path, abnormal_data_path, normal_label, abnormal_label,
                                          seq_length, seq_step, num_signals)

    def load_normal_data(self, data_path):
        return pd.read_csv(data_path, skiprows=4).drop(self.drop_columns, axis=1).dropna(axis=0).values[21600:]

    def load_abnormal_data(self, data_path):
        return pd.read_csv(data_path).drop(self.drop_columns, axis=1).dropna(axis=0).values[21600:]


class MnistDataset(Data):
    def __init__(self, normal_data_path, abnormal_data_path, normal_label, abnormal_label, seq_length=None,
                 seq_step=None, num_signals=None):
        super(MnistDataset, self).__init__(normal_data_path, abnormal_data_path, normal_label, abnormal_label,
                                           seq_length, seq_step, num_signals)

    def load_normal_data(self, data_path):
        with open(data_path, 'rb') as f:
            samples, labels = pickle.load(f)
            return samples, labels

    def load_abnormal_data(self, data_path):
        with open(data_path, 'rb') as f:
            samples, labels = pickle.load(f)
            return samples, labels

    def preprocess_data(self, samples, is_normal, normal_label, abnormal_label, seq_step, seq_length, num_signals):
        samples, labels = samples[0], samples[1]
        batch_size, seq_length = samples.shape[0], samples.shape[1]

        if is_normal:
            self.scaler1 = MinMaxScaler([-1, 1]).fit(samples.flatten().reshape((-1, 1)))

        samples = self.scaler1.transform(samples.flatten().reshape((-1, 1))).reshape(batch_size, seq_length, 28, 28)
        samples = Resize((32, 32))(torch.from_numpy(samples))

        label = normal_label if is_normal else abnormal_label
        labels = np.ones((batch_size, seq_length, 1)) * label

        return samples, labels
