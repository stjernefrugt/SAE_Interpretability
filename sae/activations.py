"""
Activation Collection from GPT-2 Small

Collects residual stream activations from a specified layer of GPT-2 small
using TransformerLens hooks. Stores activations for efficient SAE training.

Analogous to collecting X-ray scattering patterns: we run many "exposures"
(forward passes on text) and collect the resulting "diffraction patterns"
(activation vectors) for subsequent decomposition analysis.
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from transformer_lens import HookedTransformer


class ActivationCollector:
    """
    Collect residual stream activations from GPT-2 small.

    Args:
        model_name: TransformerLens model name (default: "gpt2-small")
        layer: Which residual stream layer to hook (default: 6)
        device: torch device
    """

    def __init__(
        self,
        model_name: str = "gpt2-small",
        layer: int = 6,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model_name = model_name
        self.layer = layer
        self.device = device
        self.model = None

    def load_model(self):
        """Load the transformer model."""
        print(f"Loading {self.model_name}...")
        self.model = HookedTransformer.from_pretrained(
            self.model_name, device=self.device
        )
        self.model.eval()
        self.d_model = self.model.cfg.d_model
        print(f"Loaded. d_model={self.d_model}, n_layers={self.model.cfg.n_layers}")
        return self

    @property
    def hook_name(self) -> str:
        """The TransformerLens hook point name for residual stream at our layer."""
        return f"blocks.{self.layer}.hook_resid_post"

    def collect_from_dataset(
        self,
        n_tokens: int = 1_000_000,
        dataset_name: str = "monology/pile-uncopyrighted",
        dataset_split: str = "train",
        seq_len: int = 128,
        batch_size: int = 32,
        save_path: str | None = None,
    ) -> torch.Tensor:
        """
        Collect activations by running the model on a text dataset.

        Args:
            n_tokens: Total number of token positions to collect activations from
            dataset_name: HuggingFace dataset name
            dataset_split: Dataset split
            seq_len: Sequence length for tokenization
            batch_size: Batch size for forward passes
            save_path: Optional path to save collected activations

        Returns:
            Tensor of shape (n_tokens, d_model) with collected activations
        """
        if self.model is None:
            self.load_model()

        # Load and tokenize dataset
        print(f"Loading dataset: {dataset_name}...")
        dataset = load_dataset(dataset_name, split=dataset_split, streaming=True)

        all_activations = []
        tokens_collected = 0
        n_sequences = n_tokens // seq_len

        # Tokenize and batch
        token_batches = self._tokenize_streaming(
            dataset, seq_len=seq_len, n_sequences=n_sequences
        )

        print(f"Collecting activations from layer {self.layer} ({self.hook_name})...")
        pbar = tqdm(total=n_tokens, desc="Tokens collected")

        for batch_tokens in self._batchify(token_batches, batch_size):
            batch_tokens = batch_tokens.to(self.device)

            # Run model with hook to capture activations
            with torch.no_grad():
                _, cache = self.model.run_with_cache(
                    batch_tokens,
                    names_filter=self.hook_name,
                )

            # Extract activations: shape (batch, seq_len, d_model)
            acts = cache[self.hook_name]
            # Flatten to (batch * seq_len, d_model)
            acts = acts.reshape(-1, self.d_model).cpu()
            all_activations.append(acts)

            tokens_collected += acts.shape[0]
            pbar.update(acts.shape[0])

            if tokens_collected >= n_tokens:
                break

        pbar.close()

        # Concatenate and trim to exact size
        activations = torch.cat(all_activations, dim=0)[:n_tokens]
        print(f"Collected {activations.shape[0]} activation vectors of dim {activations.shape[1]}")

        # Optionally save
        if save_path:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(activations, path)
            print(f"Saved to {path}")

        return activations

    def _tokenize_streaming(self, dataset, seq_len: int, n_sequences: int):
        """Tokenize streaming dataset into fixed-length sequences."""
        token_buffer = []
        sequences_yielded = 0

        for example in dataset:
            text = example.get("text", "")
            if not text:
                continue

            tokens = self.model.to_tokens(text, prepend_bos=False).squeeze(0)
            token_buffer.extend(tokens.tolist())

            while len(token_buffer) >= seq_len:
                yield torch.tensor(token_buffer[:seq_len])
                token_buffer = token_buffer[seq_len:]
                sequences_yielded += 1

                if sequences_yielded >= n_sequences:
                    return

    def _batchify(self, token_iter, batch_size: int):
        """Group token sequences into batches."""
        batch = []
        for tokens in token_iter:
            batch.append(tokens)
            if len(batch) == batch_size:
                yield torch.stack(batch)
                batch = []
        if batch:
            yield torch.stack(batch)


def load_activations(path: str) -> torch.Tensor:
    """Load previously saved activations."""
    return torch.load(path, weights_only=True)
