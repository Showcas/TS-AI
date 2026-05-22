from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def vae_loss(
    tensor_batch: torch.Tensor,
    reconstructed_tensor_batch: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the VAE loss as the sum of reconstruction loss and KL divergence.
    Args:
        tensor_batch: Original input tensor batch.
        reconstructed_tensor_batch: Reconstructed tensor batch from the VAE.
        mu: Mean from the encoder's latent space.
        logvar: Log variance from the encoder's latent space.
    Returns:
        total_loss: Total VAE loss.
    """
    recon = F.mse_loss(
        input=reconstructed_tensor_batch, target=tensor_batch, reduction="sum"
    )
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon + kl
    return total_loss, recon, kl


class ConvEncoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),  # 28 → 14
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 14 → 7
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.conv(x)
        return self.fc_mu(h), self.fc_logvar(h)


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 7 * 7)
        self.deconv = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 7 → 14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),  # 14 → 28
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, 64, 7, 7)
        return self.deconv(h)


def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """
    Reparameterization trick to sample from N(mu, var) from N(0,1).
    Args:
        mu: Mean from the encoder's latent space.
        log_var: Log variance from the encoder's latent space.
    Returns:
        Sampled latent vector.
    """
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std


class ConvVAE(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.encoder = ConvEncoder(latent_dim=latent_dim)
        self.decoder = ConvDecoder(latent_dim=latent_dim)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar
