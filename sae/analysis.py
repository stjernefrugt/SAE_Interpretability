"""
Feature Analysis and SVD Comparison

Tools for interpreting learned SAE features and comparing them with SVD
decomposition — drawing an explicit connection between traditional linear
decomposition methods (as used in X-ray scattering analysis) and the
sparse overcomplete decomposition that SAEs provide.

The key insight: SVD finds orthogonal directions that maximize variance
(analogous to principal temporal/structural modes in time-resolved X-ray
scattering). SAEs find sparse, overcomplete directions that maximize
interpretability. Both are decomposition methods for superposed signals,
but SAEs better capture the sparse, non-orthogonal nature of neural
computation.
"""

import torch
import numpy as np
from torch import Tensor
from transformer_lens import HookedTransformer
from typing import Optional


class FeatureAnalyzer:
    """
    Analyze features learned by a Sparse Autoencoder.

    Provides tools to find maximally activating examples, characterize
    features, and compare with SVD decomposition.
    """

    def __init__(self, sae, model: Optional[HookedTransformer] = None):
        """
        Args:
            sae: Trained SparseAutoencoder
            model: Optional HookedTransformer for decoding tokens
        """
        self.sae = sae
        self.model = model

    @torch.no_grad()
    def get_feature_activations(self, activations: Tensor) -> Tensor:
        """Get SAE feature activations for a batch of model activations."""
        self.sae.eval()
        return self.sae.encode(activations.to(next(self.sae.parameters()).device))

    @torch.no_grad()
    def top_activating_examples(
        self, feature_idx: int, activations: Tensor, k: int = 10
    ) -> dict:
        """
        Find the top-k activation vectors that most strongly activate a given feature.

        Args:
            feature_idx: Which SAE feature to analyze
            activations: Full activation dataset (n_tokens, d_model)
            k: Number of top examples to return

        Returns:
            Dict with 'indices', 'values' of top activating positions
        """
        feat_acts = self.get_feature_activations(activations)[:, feature_idx]
        top_values, top_indices = feat_acts.topk(k)

        return {
            "indices": top_indices.cpu(),
            "values": top_values.cpu(),
        }

    @torch.no_grad()
    def feature_statistics(self, activations: Tensor) -> dict:
        """
        Compute statistics for all features across a dataset.

        Returns:
            Dict with per-feature statistics:
            - activation_freq: fraction of inputs where feature is active
            - mean_activation: mean activation when active
            - max_activation: maximum activation value
        """
        feat_acts = self.get_feature_activations(activations)
        n_samples = feat_acts.shape[0]

        is_active = feat_acts > 0
        activation_freq = is_active.float().mean(dim=0)

        # Mean activation conditioned on being active
        sum_when_active = (feat_acts * is_active.float()).sum(dim=0)
        count_active = is_active.float().sum(dim=0).clamp(min=1)
        mean_activation = sum_when_active / count_active

        max_activation = feat_acts.max(dim=0).values

        return {
            "activation_freq": activation_freq.cpu(),
            "mean_activation": mean_activation.cpu(),
            "max_activation": max_activation.cpu(),
            "n_dead": (activation_freq == 0).sum().item(),
            "n_features": feat_acts.shape[1],
        }

    def display_top_examples(
        self,
        feature_idx: int,
        activations: Tensor,
        tokens_flat: Tensor,
        seq_len: int,
        k: int = 5,
        context_window: int = 10,
    ) -> list[dict]:
        """
        Find and format top activating examples with surrounding context.

        Args:
            feature_idx: SAE feature index
            activations: Activation tensor (n_tokens, d_model)
            tokens_flat: Flattened token tensor (n_tokens,)
            seq_len: Original sequence length (for finding position in sequence)
            k: Number of examples
            context_window: Tokens of context before/after

        Returns:
            List of dicts with 'text', 'target_token', 'activation_value'
        """
        if self.model is None:
            raise ValueError("Need a model for token decoding. Pass model to __init__.")

        top = self.top_activating_examples(feature_idx, activations, k=k)
        results = []

        for idx, val in zip(top["indices"], top["values"]):
            idx = idx.item()
            start = max(0, idx - context_window)
            end = min(len(tokens_flat), idx + context_window + 1)

            context_tokens = tokens_flat[start:end]
            target_pos = idx - start

            # Decode tokens
            token_strs = [self.model.tokenizer.decode(t.item()) for t in context_tokens]

            results.append({
                "tokens": token_strs,
                "target_pos": target_pos,
                "target_token": token_strs[target_pos],
                "activation_value": val.item(),
                "global_idx": idx,
            })

        return results


def svd_comparison(activations: Tensor, n_components: int = 50) -> dict:
    """
    Perform SVD decomposition on activations for comparison with SAE.

    This is the classic approach used in X-ray scattering analysis:
    decompose a data matrix into orthogonal components ordered by
    variance explained. In time-resolved X-ray scattering, the left
    singular vectors are temporal profiles and the right singular
    vectors are structural modes.

    For neural network activations:
    - Left singular vectors: how each token position projects onto components
    - Right singular vectors: directions in activation space
    - Singular values: variance explained by each component

    The limitation vs SAEs: SVD enforces orthogonality, which means each
    component typically mixes multiple semantic concepts. SAEs allow
    non-orthogonal, sparse features that tend to be monosemantic.

    Args:
        activations: (n_tokens, d_model) activation matrix
        n_components: Number of SVD components to compute

    Returns:
        Dict with:
        - U: left singular vectors (n_tokens, n_components)
        - S: singular values (n_components,)
        - Vt: right singular vectors / directions (n_components, d_model)
        - explained_variance_ratio: fraction of total variance per component
    """
    # Center the data (standard for PCA/SVD analysis)
    mean = activations.mean(dim=0, keepdim=True)
    centered = activations - mean

    # Compute truncated SVD
    U, S, Vt = torch.linalg.svd(centered, full_matrices=False)

    # Truncate to n_components
    U = U[:, :n_components]
    S = S[:n_components]
    Vt = Vt[:n_components, :]

    # Explained variance ratio
    total_variance = (S ** 2).sum()
    explained_variance_ratio = (S ** 2) / total_variance

    return {
        "U": U.cpu(),
        "S": S.cpu(),
        "Vt": Vt.cpu(),
        "mean": mean.cpu(),
        "explained_variance_ratio": explained_variance_ratio.cpu(),
    }


def svd_top_activating_examples(
    svd_result: dict,
    component_idx: int,
    k: int = 10,
) -> dict:
    """
    Find tokens that project most strongly onto a given SVD component.

    Analogous to finding time-points with strongest projection onto a
    structural mode in X-ray scattering SVD analysis.
    """
    projections = svd_result["U"][:, component_idx] * svd_result["S"][component_idx]
    top_values, top_indices = projections.abs().topk(k)
    signs = projections[top_indices].sign()

    return {
        "indices": top_indices,
        "values": top_values * signs,
    }


def compare_interpretability(
    sae_examples: list[dict],
    svd_examples: list[dict],
) -> str:
    """Format a side-by-side comparison of SAE feature vs SVD component examples."""
    lines = []
    lines.append("=" * 70)
    lines.append("SAE FEATURE (sparse, overcomplete)")
    lines.append("-" * 70)
    for ex in sae_examples[:5]:
        tokens = ex["tokens"]
        target = ex["target_pos"]
        display = ""
        for i, t in enumerate(tokens):
            if i == target:
                display += f">>>{t}<<<"
            else:
                display += t
        lines.append(f"  [{ex['activation_value']:.2f}] {display.strip()}")

    lines.append("")
    lines.append("=" * 70)
    lines.append("SVD COMPONENT (orthogonal, variance-maximizing)")
    lines.append("-" * 70)
    for ex in svd_examples[:5]:
        tokens = ex["tokens"]
        target = ex["target_pos"]
        display = ""
        for i, t in enumerate(tokens):
            if i == target:
                display += f">>>{t}<<<"
            else:
                display += t
        lines.append(f"  [{ex['activation_value']:.2f}] {display.strip()}")

    lines.append("=" * 70)
    return "\n".join(lines)


@torch.no_grad()
def logit_lens(
    sae,
    model,
    feature_indices: list[int] | None = None,
    top_k: int = 15,
) -> dict:
    """
    Project SAE decoder directions onto the model's unembedding matrix.

    For each feature, this reveals what tokens the feature "wants to predict"
    by computing: logits_i = W_dec[i] @ W_U

    This is the most scalable method for semantic feature assignment — it
    runs in O(d_sae * vocab_size) and requires no dataset examples.

    Args:
        sae: Trained SparseAutoencoder
        model: HookedTransformer model (provides W_U unembedding)
        feature_indices: Which features to analyze (default: all alive features)
        top_k: Number of top/bottom tokens to return per feature

    Returns:
        Dict mapping feature_idx → {top_tokens, top_logits, bottom_tokens, bottom_logits}
    """
    # Get unembedding matrix: (d_model, vocab_size)
    W_U = model.W_U  # shape: (d_model, d_vocab)

    # Get decoder weights: (d_sae, d_model)
    W_dec = sae.W_dec.data.cpu()

    if feature_indices is None:
        feature_indices = list(range(sae.d_sae))

    results = {}
    for feat_idx in feature_indices:
        # Project decoder direction onto vocabulary
        # decoder_dir: (d_model,) @ W_U: (d_model, d_vocab) → (d_vocab,)
        decoder_dir = W_dec[feat_idx].to(W_U.device)
        logits = decoder_dir @ W_U  # (d_vocab,)

        # Top tokens (feature promotes these)
        top_vals, top_ids = logits.topk(top_k)
        top_tokens = [model.tokenizer.decode(t.item()) for t in top_ids]

        # Bottom tokens (feature suppresses these)
        bot_vals, bot_ids = logits.topk(top_k, largest=False)
        bot_tokens = [model.tokenizer.decode(t.item()) for t in bot_ids]

        results[feat_idx] = {
            "top_tokens": top_tokens,
            "top_logits": top_vals.cpu(),
            "bottom_tokens": bot_tokens,
            "bottom_logits": bot_vals.cpu(),
        }

    return results


@torch.no_grad()
def activation_patching(
    sae,
    model,
    text: str,
    feature_idx: int,
    layer: int = 6,
) -> dict:
    """
    Measure the causal effect of a single SAE feature via activation patching.

    Method:
    1. Run model normally, get output logits
    2. Run model with hook: at the target layer, replace activation with
       SAE reconstruction MINUS the target feature
    3. Measure KL divergence between original and patched outputs

    This gives causal evidence of what a feature does — not just correlation.

    Args:
        sae: Trained SparseAutoencoder
        model: HookedTransformer
        text: Input text to analyze
        feature_idx: Which SAE feature to ablate
        layer: Which layer the SAE was trained on

    Returns:
        Dict with kl_divergence per position, original/patched top predictions
    """
    import torch.nn.functional as F

    device = next(model.parameters()).device
    tokens = model.to_tokens(text).to(device)

    hook_name = f"blocks.{layer}.hook_resid_post"

    # Run normally
    original_logits = model(tokens)  # (1, seq_len, vocab)

    # Run with feature ablation hook
    def ablate_feature_hook(activation, hook):
        # activation shape: (batch, seq_len, d_model)
        batch_size, seq_len, d_model = activation.shape
        flat = activation.reshape(-1, d_model)

        # Encode with SAE
        features = sae.encode(flat)
        # Zero out target feature
        features[:, feature_idx] = 0.0
        # Reconstruct without this feature
        reconstructed = sae.decode(features)

        return reconstructed.reshape(batch_size, seq_len, d_model)

    patched_logits = model.run_with_hooks(
        tokens,
        fwd_hooks=[(hook_name, ablate_feature_hook)],
    )

    # KL divergence per position
    original_probs = F.softmax(original_logits[0], dim=-1)
    patched_probs = F.softmax(patched_logits[0], dim=-1)
    kl_div = (original_probs * (original_probs.log() - patched_probs.log())).sum(dim=-1)

    # Top predictions at highest-KL position
    max_kl_pos = kl_div.argmax().item()
    orig_top5 = original_probs[max_kl_pos].topk(5)
    patched_top5 = patched_probs[max_kl_pos].topk(5)

    token_strs = [model.tokenizer.decode(t.item()) for t in tokens[0]]

    return {
        "kl_divergence": kl_div.cpu(),
        "max_kl_position": max_kl_pos,
        "max_kl_token": token_strs[max_kl_pos] if max_kl_pos < len(token_strs) else "",
        "tokens": token_strs,
        "original_top5": {
            "tokens": [model.tokenizer.decode(t.item()) for t in orig_top5.indices],
            "probs": orig_top5.values.cpu(),
        },
        "patched_top5": {
            "tokens": [model.tokenizer.decode(t.item()) for t in patched_top5.indices],
            "probs": patched_top5.values.cpu(),
        },
    }
