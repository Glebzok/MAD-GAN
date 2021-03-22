import torch
import torch.nn as nn
torch.use_deterministic_algorithms(True)


class LSTMGenerator(nn.Module):

    def __init__(self, latent_space_dim, out_space_dim, lstm_layers, lstm_hidden_dim):
        super(LSTMGenerator, self).__init__()

        self.out_space_dim = out_space_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers

        self.lstm = nn.LSTM(input_size=latent_space_dim, hidden_size=lstm_hidden_dim, num_layers=lstm_layers,
                            batch_first=True)
        self.linear = nn.Linear(in_features=lstm_hidden_dim, out_features=out_space_dim)

    def forward(self, input):
        batch_size, seq_len = input.size(0), input.size(1)

        recurrent_features, _ = self.lstm(input)
        outputs = self.linear(
            recurrent_features.contiguous().view(batch_size * seq_len, self.lstm_hidden_dim))  # no tanh
        outputs = outputs.view(batch_size, seq_len, self.out_space_dim)

        return outputs


class LSTMDiscriminator(nn.Module):

    def __init__(self, out_space_dim, lstm_layers, lstm_hidden_dim):
        super(LSTMDiscriminator, self).__init__()

        self.lstm_hidden_dim = lstm_hidden_dim

        self.lstm = nn.LSTM(input_size=out_space_dim, hidden_size=lstm_hidden_dim, num_layers=lstm_layers,
                            batch_first=True)
        self.linear = nn.Linear(in_features=lstm_hidden_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        batch_size, seq_len = input.size(0), input.size(1)

        recurrent_features, _ = self.lstm(input)
        outputs = self.sigmoid(
            self.linear(recurrent_features.contiguous().view(batch_size * seq_len, self.lstm_hidden_dim)))

        outputs = outputs.view(batch_size, seq_len, 1)

        return outputs


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class CNN_LSTMGenerator(nn.Module):

    def __init__(self, latent_space_dim, lstm_hidden_dim, lstm_layers, cnn_features):
        super(CNN_LSTMGenerator, self).__init__()

        self.lstm_hidden_dim = lstm_hidden_dim

        self.lstm = nn.LSTM(input_size=latent_space_dim, hidden_size=lstm_hidden_dim, num_layers=lstm_layers,
                            batch_first=True)

        self.cnn = nn.Sequential(
            # state size. (lstm_features) x 2 x 2
            nn.ConvTranspose2d(lstm_hidden_dim, cnn_features * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(cnn_features * 4),
            nn.ReLU(True),
            # state size. (cnn_features*4) x 4 x 4
            nn.ConvTranspose2d(cnn_features * 4, cnn_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(cnn_features * 2),
            nn.ReLU(True),
            # state size. (cnn_features*2) x 8 x 8
            nn.ConvTranspose2d(cnn_features * 2, cnn_features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(cnn_features),
            nn.ReLU(True),
            # state size. (cnn_features) x 16 x 16
            nn.ConvTranspose2d(cnn_features, 1, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. 1 x 32 x 32
        )

    def forward(self, input):
        batch_size, seq_len = input.size(0), input.size(1)

        recurrent_features, _ = self.lstm(input)
        recurrent_features = recurrent_features.contiguous().view(batch_size * seq_len, self.lstm_hidden_dim, 1, 1)
        outputs = self.cnn(recurrent_features).view(batch_size, seq_len, 32, 32)

        return outputs


class CNN_LSTMDiscriminator(nn.Module):

    def __init__(self, lstm_hidden_dim, lstm_layers, cnn_features):
        super(CNN_LSTMDiscriminator, self).__init__()

        self.lstm_hidden_dim = lstm_hidden_dim

        self.lstm = nn.LSTM(input_size=cnn_features * 4, hidden_size=lstm_hidden_dim, num_layers=lstm_layers,
                            batch_first=True)
        self.cnn = nn.Sequential(
            # input is 1 x 64 x 64
            nn.Conv2d(1, cnn_features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. cnn_features x 32 x 32
            nn.Conv2d(cnn_features, cnn_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(cnn_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (cnn_features*2) x 16 x 16
            nn.Conv2d(cnn_features * 2, cnn_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(cnn_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (cnn_features*4) x 8 x 8
            nn.Conv2d(cnn_features * 4, cnn_features * 4, 4, 1, 0, bias=False),
        )
        self.linear = nn.Linear(in_features=lstm_hidden_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        batch_size, seq_len = input.size(0), input.size(1)

        convolutional_features = self.cnn(input.view(batch_size * seq_len, 1, 32, 32)).view(batch_size, seq_len, -1)

        recurrent_features, _ = self.lstm(convolutional_features)
        outputs = self.sigmoid(
            self.linear(recurrent_features.contiguous().view(batch_size * seq_len, self.lstm_hidden_dim)))

        outputs = outputs.view(batch_size, seq_len, 1)

        return outputs
