import torch
import torch.nn as nn
from torch.autograd import Variable

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Stacked_VAE(nn.Module):
    def __init__(self, n_in: int = 784, n_hidden: int = 400, n_latent: int = 2, n_layers: int = 1):
        """
        Initialize a Stacked Variational Autoencoder (VAE).

        Args:
            n_in (int): Number of input features.
            n_hidden (int): Number of hidden units in the neural network layers.
            n_latent (int): Number of latent variables.
            n_layers (int): Number of layers in the encoder and decoder networks.
        """
        super(Stacked_VAE, self).__init__()
        self.soft_zero = 1e-10
        self.n_latent = n_latent
        self.n_in = n_in
        self.mu = None

        # Encoder layers
        encoder_layers = []
        for i in range(n_layers):
            in_features = n_in if i == 0 else n_hidden
            out_features = n_latent * 2 if i == n_layers - 1 else n_hidden
            encoder_layers.append(nn.Linear(in_features, out_features))
            if i < n_layers - 1:
                encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder layers
        decoder_layers = []
        for i in range(n_layers):
            in_features = n_latent if i == 0 else n_hidden
            out_features = n_in if i == n_layers - 1 else n_hidden
            decoder_layers.append(nn.Linear(in_features, out_features))
            if i < n_layers - 1:
                decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

    def encoded(self, x: torch.Tensor) -> tuple:
        """
        Encode the input data.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            tuple: Tuple containing the mean (mu) and log variance (lv) of the latent space.
        """
        h = self.encoder(x)
        mu_lv = torch.split(h, self.n_latent, dim=1)
        return mu_lv[0], mu_lv[1]

    def decoded(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode the latent space representation.

        Args:
            z (torch.Tensor): Latent space representation.

        Returns:
            torch.Tensor: Decoded output.
        """
        return self.decoder(z)

    def reparameterize(self, mu: torch.Tensor, lv: torch.Tensor) -> torch.Tensor:
        """
        Reparameterize the latent space.

        Args:
            mu (torch.Tensor): Mean of the latent space.
            lv (torch.Tensor): Log variance of the latent space.

        Returns:
            torch.Tensor: Reparameterized latent space.
        """
        eps = torch.randn_like(mu)
        z = mu + torch.exp(0.5 * lv) * eps
        return z

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass through the VAE.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            tuple: Tuple containing the reconstructed output (y) and the weight loss.
        """
        mu, lv = self.encoded(x)
        z = self.reparameterize(mu, lv)
        y = self.decoded(z)

        # Compute the loss components
        KL = 0.5 * torch.sum(1 + lv - mu * mu - lv.exp(), dim=1)
        logloss = torch.sum(x * torch.log(y + self.soft_zero) + (1 - x) * torch.log(1 - y + self.soft_zero), dim=1)
        loss = -logloss - KL

        weight_loss = loss.unsqueeze(1)

        return y, weight_loss

class BiLSTM(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 100, n_layers: int = 5, 
                 dropout: float = 0.5, output_size: int = 15, device=device):
        """
      Initialize a Bidirectional LSTM (BiLSTM) model.

      Args:
          input_size (int): Number of input features.
          hidden_size (int): Number of hidden units in the LSTM layers.
          n_layers (int): Number of LSTM layers.
          dropout (float): Dropout probability.
          output_size (int): Number of output units.
        """
        super(BiLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = n_layers, bidirectional = True, dropout = dropout, batch_first = True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.relu = nn.ReLU()
        self.device = device

    def forward(self, x):
        """
        Forward pass through the BiLSTM.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Output predictions.
        """
        h0 = Variable(torch.zeros(self.n_layers * 2, x.size(0), self.hidden_size)).to(device)
        c0 = Variable(torch.zeros(self.n_layers * 2, x.size(0), self.hidden_size)).to(device)
        output, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(output[:, -self.output_size, :])
        out = self.relu(out)
        return out