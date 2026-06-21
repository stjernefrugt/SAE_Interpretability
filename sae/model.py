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
