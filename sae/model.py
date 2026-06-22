"""
Sparse Autoencoder for Mechanistic Interpretability

A sparse autoencoder learns an overcomplete set of features from neural network
activations. The key insight (from Anthropic's "Towards Monosemanticity") is that
neural networks represent many more concepts than they have dimensions — features
are superposed. SAEs decompose these superposed activations into sparse,
interpretable directions.

This is directly analogous to decomposing superposed X-ray scattering signals
into individual structural/dynamical components — except here the "features" are
computational motifs rather than physical modes.

Architecture:
    encoder: x → ReLU(W_enc @ (x - b_dec) + b_enc)
    decoder: f → W_dec @ f + b_dec
    loss: MSE(x, x_hat) + l1_coeff * |f|_1

The decoder columns are kept at unit norm (following Anthropic's approach), which
ensures the L1 penalty acts on meaningful feature activation magnitudes.
"""

import torch
import torch.nn as nn
from torch import Tensor


class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder for decomposing neural network activations.

    Args:
        d_model: Dimension of the input activations (e.g., 768 for GPT-2 small)
        d_sae: Number of SAE features (overcomplete: d_sae >> d_model)
        l1_coeff: Weight of L1 sparsity penalty
    """

    def __init__(self, d_model: int, d_sae: int, l1_coeff: float = 5e-3):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.l1_coeff = l1_coeff

        # Encoder weights and bias
        self.W_enc = nn.Parameter(torch.empty(d_model, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))

        # Decoder weights and bias (bias is shared as the "centering" term)
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize with Kaiming uniform, then normalize decoder columns."""
        nn.init.kaiming_uniform_(self.W_enc)
        nn.init.kaiming_uniform_(self.W_dec)
        # Normalize decoder columns to unit norm
        with torch.no_grad():
            self.W_dec.data = self.W_dec.data / self.W_dec.data.norm(dim=1, keepdim=True)

    def encode(self, x: Tensor) -> Tensor:
        """Encode activations into sparse feature space."""
        # Subtract decoder bias (centering), then project and apply ReLU
        return torch.relu(
            (x - self.b_dec) @ self.W_enc + self.b_enc
        )

    def decode(self, f: Tensor) -> Tensor:
        """Reconstruct activations from sparse features."""
        return f @ self.W_dec + self.b_dec

    def forward(self, x: Tensor) -> dict:
        """
        Forward pass: encode, decode, compute losses.

        Returns dict with:
            x_hat: reconstructed activations
            features: sparse feature activations
            loss: total loss (reconstruction + sparsity)
            mse_loss: reconstruction loss
            l1_loss: sparsity loss
            l0: average number of active features (for monitoring)
        """
        features = self.encode(x)
        x_hat = self.decode(features)

        # Reconstruction loss (MSE)
        mse_loss = (x - x_hat).pow(2).mean()

        # Sparsity loss (L1 on feature activations)
        l1_loss = features.abs().mean()

        # Total loss
        loss = mse_loss + self.l1_coeff * l1_loss

        # L0: average number of active features per sample (for monitoring only)
        l0 = (features > 0).float().sum(dim=-1).mean()

        return {
            "x_hat": x_hat,
            "features": features,
            "loss": loss,
            "mse_loss": mse_loss,
            "l1_loss": l1_loss,
            "l0": l0,
        }

    @torch.no_grad()
    def normalize_decoder(self):
        """Normalize decoder weight columns to unit norm (call after each optimizer step)."""
        self.W_dec.data = self.W_dec.data / self.W_dec.data.norm(dim=1, keepdim=True)

    @torch.no_grad()
    def get_dead_features(self, feature_activations: Tensor, threshold: float = 0.0) -> Tensor:
        """Identify features that never activate (dead features)."""
        max_activations = feature_activations.max(dim=0).values
        return max_activations <= threshold


class MatryoshkaSAE(nn.Module):
    """
    Matryoshka Sparse Autoencoder — features are ordered by importance.

    Inspired by Matryoshka Representation Learning: the SAE is trained so that
    the first k features form a useful reconstruction on their own, for multiple
    values of k. This creates a natural hierarchy from coarse to fine features.

    Analogy: In X-ray scattering SVD, singular values naturally order components
    by importance. Standard SAEs lose this ordering. Matryoshka SAEs recover it
    while keeping the benefits of sparse overcomplete representations.

    Training loss:
        L = Σ_k w_k * MSE(x, decode(features[:k])) + l1_coeff * L1(features)

    where k ranges over truncation_points.

    Args:
        d_model: Dimension of input activations
        d_sae: Total number of SAE features
        l1_coeff: L1 sparsity penalty weight
        truncation_points: List of prefix sizes at which to evaluate reconstruction
    """

    def __init__(
        self,
        d_model: int,
        d_sae: int,
        l1_coeff: float = 5e-3,
        truncation_points: list[int] | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.l1_coeff = l1_coeff

        if truncation_points is None:
            # Default: logarithmic spacing
            truncation_points = [d_sae // 2**i for i in range(4, -1, -1)]
            truncation_points = [k for k in truncation_points if k > 0]
        self.truncation_points = sorted(truncation_points)

        # Encoder
        self.W_enc = nn.Parameter(torch.empty(d_model, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))

        # Decoder
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.W_enc)
        nn.init.kaiming_uniform_(self.W_dec)
        with torch.no_grad():
            self.W_dec.data = self.W_dec.data / self.W_dec.data.norm(dim=1, keepdim=True)

    def encode(self, x: Tensor) -> Tensor:
        return torch.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    def decode(self, f: Tensor) -> Tensor:
        return f @ self.W_dec + self.b_dec

    def decode_prefix(self, f: Tensor, k: int) -> Tensor:
        """Decode using only the first k features."""
        return f[:, :k] @ self.W_dec[:k, :] + self.b_dec

    def forward(self, x: Tensor) -> dict:
        """
        Forward pass with Matryoshka multi-scale reconstruction losses.

        Returns dict with:
            x_hat: full reconstruction
            features: sparse feature activations
            loss: total loss (sum of prefix losses + sparsity)
            mse_loss: MSE at full d_sae (for comparison)
            prefix_losses: dict mapping k → MSE at that prefix
            l0: average number of active features
        """
        features = self.encode(x)

        # Full reconstruction
        x_hat = self.decode(features)
        mse_loss = (x - x_hat).pow(2).mean()

        # Matryoshka loss: reconstruction at each truncation point
        prefix_losses = {}
        total_recon_loss = torch.tensor(0.0, device=x.device)

        for k in self.truncation_points:
            x_hat_k = self.decode_prefix(features, k)
            loss_k = (x - x_hat_k).pow(2).mean()
            prefix_losses[k] = loss_k.item()
            total_recon_loss = total_recon_loss + loss_k

        # Average over truncation points
        total_recon_loss = total_recon_loss / len(self.truncation_points)

        # Sparsity
        l1_loss = features.abs().mean()
        loss = total_recon_loss + self.l1_coeff * l1_loss

        l0 = (features > 0).float().sum(dim=-1).mean()

        return {
            "x_hat": x_hat,
            "features": features,
            "loss": loss,
            "mse_loss": mse_loss,
            "l1_loss": l1_loss,
            "prefix_losses": prefix_losses,
            "l0": l0,
        }

    @torch.no_grad()
    def normalize_decoder(self):
        self.W_dec.data = self.W_dec.data / self.W_dec.data.norm(dim=1, keepdim=True)
