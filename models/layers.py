import torch
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class VariationalEncoder(nn.Module):
    def __init__(self, num_input_channels,
                 base_channel_size,
                 latent_dims,
                 ):
        super(VariationalEncoder, self).__init__()
        c_hid = base_channel_size
        self.conv1 = nn.Conv2d(num_input_channels, c_hid, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, stride=2, padding=1)
        # self.batch2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(2 * c_hid, 2 * 2 * c_hid, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(2 * 2 * c_hid, 2 * 2 * 2 * c_hid, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(2 * 2 * 2 * c_hid, 2 * 2 * 2 * 2 * c_hid, kernel_size=3, padding=1, stride=2)
        self.linear1 = nn.Linear(4 * 4 * 16 * c_hid, 4 * 2 * c_hid)
        self.linear2 = nn.Linear(4 * 2 * c_hid, latent_dims)
        self.linear3 = nn.Linear(4 * 2 * c_hid, latent_dims)
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()  # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = torch.flatten(x, start_dim=1)
        x = self.linear1(x)
        x = F.relu(x)
        mu = self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1 / 2).sum()
        return z
        # return x


class Decoder(nn.Module):

    def __init__(self, num_input_channels,
                 base_channel_size,
                 latent_dims):
        super().__init__()
        c_hid = base_channel_size
        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dims, 4 * 2 * c_hid),
            nn.ReLU(True),
            nn.Linear(4 * 2 * c_hid, 4 * 4 * 16 * c_hid),
            nn.ReLU(True)
        )

        # self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 3, 3))
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(2 * 2 * 2 * 2 * c_hid, 2 * 2 * 2 * c_hid, kernel_size=3, output_padding=1, padding=1,
                               stride=2),
            # nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(2 * 2 * 2 * c_hid, 2 * 2 * c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(2 * 2 * c_hid, 2 * c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),
            # nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(2 * c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        # x = self.unflatten(x)
        x = x.reshape(x.shape[0], -1, 4, 4)
        x = self.decoder_conv(x)
        # x = torch.tanh(x)
        return x