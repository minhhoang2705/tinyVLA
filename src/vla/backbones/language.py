"""Language backbone for instruction encoding.

Provides wrappers for pretrained language models (GPT-2, BERT, etc.) to encode
text instructions into embeddings for VLA models. Supports multiple output modes
and optional projection layers.

Example:
    >>> from vla.backbones import GPT2Backbone
    >>> encoder = GPT2Backbone(model_name="gpt2", frozen=True)
    >>> texts = ["pick up the red block", "place it on the table"]
    >>> embeddings = encoder(texts=texts)  # [B, 1, D]
"""

from typing import Any, Dict, List, Literal, Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, GPT2Model, GPT2Tokenizer

from vla.registry import LANGUAGE_REGISTRY


@LANGUAGE_REGISTRY.register("gpt2")
class GPT2Backbone(nn.Module):
    """GPT-2 language model for instruction encoding.

    Wraps HuggingFace GPT-2 models with support for different output modes
    (last token, mean pooling, all tokens) and optional projection layers.
    Typically used frozen for transfer learning in VLA models.

    Args:
        model_name: HuggingFace model name (e.g., "gpt2", "gpt2-medium")
        frozen: Freeze model parameters (no gradients)
        output_mode: Output extraction mode:
            - "last": Last non-padded token embedding [B, 1, D]
            - "mean": Mean pool over all non-padded tokens [B, 1, D]
            - "all": All token embeddings [B, L, D]
        max_length: Maximum sequence length for truncation
        proj_dim: Optional projection dimension (adds Linear layer)

    Example:
        >>> encoder = GPT2Backbone(model_name="gpt2", output_mode="mean")
        >>> embeddings = encoder(texts=["pick up block"])
        >>> print(embeddings.shape)
        torch.Size([1, 1, 768])
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        frozen: bool = True,
        output_mode: Literal["last", "mean", "all"] = "mean",
        max_length: int = 77,
        proj_dim: Optional[int] = None,
    ):
        super().__init__()
        self.output_mode = output_mode
        self.max_length = max_length

        # Load model and tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2Model.from_pretrained(model_name)

        # GPT-2 tokenizer needs pad token
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.embed_dim = self.model.config.hidden_size

        if frozen:
            self._freeze_model()

        # Optional projection
        self.proj = None
        if proj_dim is not None:
            self.proj = nn.Linear(self.embed_dim, proj_dim)
            self.embed_dim = proj_dim

    def _freeze_model(self) -> None:
        """Freeze all model parameters to prevent gradient updates."""
        for param in self.model.parameters():
            param.requires_grad = False

    def tokenize(
        self,
        texts: List[str],
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """Tokenize batch of texts.

        Args:
            texts: List of instruction strings
            device: Target device for tensors

        Returns:
            Dictionary with keys: input_ids, attention_mask
        """
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        result: Dict[str, Any] = tokens  # type: ignore
        if device:
            result = {k: v.to(device) for k, v in result.items()}
        return result  # type: ignore

    def forward(
        self,
        texts: Optional[List[str]] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode text instructions.

        Args:
            texts: List of instruction strings [B]
            input_ids: Pre-tokenized input IDs [B, L]
            attention_mask: Attention mask [B, L]

        Returns:
            Embeddings with shape:
                - [B, 1, D] if output_mode is "last" or "mean"
                - [B, L, D] if output_mode is "all"

        Raises:
            ValueError: If neither texts nor input_ids provided

        Example:
            >>> encoder = GPT2Backbone()
            >>> # From text
            >>> emb1 = encoder(texts=["pick up block"])
            >>> # From pre-tokenized
            >>> tokens = encoder.tokenize(["pick up block"])
            >>> emb2 = encoder(input_ids=tokens["input_ids"],
            ...                attention_mask=tokens["attention_mask"])
        """
        # Tokenize if needed
        if texts is not None:
            device = next(self.model.parameters()).device
            tokens = self.tokenize(texts, device)
            input_ids = tokens["input_ids"]
            attention_mask = tokens["attention_mask"]

        if input_ids is None or attention_mask is None:
            raise ValueError("Either texts or input_ids/attention_mask must be provided")

        # Forward through model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states: torch.Tensor = outputs.last_hidden_state  # type: ignore

        # Extract embeddings based on mode
        embeddings: torch.Tensor
        if self.output_mode == "last":
            # Get last non-padded token for each sequence
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
            embeddings = hidden_states[batch_idx, seq_lengths]  # [B, D]
            embeddings = embeddings.unsqueeze(1)  # [B, 1, D]
        elif self.output_mode == "mean":
            # Mean pool over non-padded tokens
            mask = attention_mask.unsqueeze(-1).float()
            embeddings = (hidden_states * mask).sum(1) / mask.sum(1)  # [B, D]
            embeddings = embeddings.unsqueeze(1)  # [B, 1, D]
        else:  # "all"
            embeddings = hidden_states  # [B, L, D]

        if self.proj is not None:
            embeddings = self.proj(embeddings)

        return embeddings


@LANGUAGE_REGISTRY.register("language_encoder")
class LanguageEncoder(nn.Module):
    """Generic language encoder supporting multiple backends.

    Provides unified interface for different language models (GPT-2, BERT, etc.)
    with automatic tokenizer/model loading and consistent output formatting.

    Args:
        backend: Backend model type or HuggingFace model name
        frozen: Freeze model parameters
        output_mode: Output extraction mode (last/mean/all)
        max_length: Maximum sequence length
        proj_dim: Optional projection dimension

    Example:
        >>> encoder = LanguageEncoder(backend="gpt2", output_mode="mean")
        >>> embeddings = encoder(texts=["test instruction"])
        >>> print(embeddings.shape)
        torch.Size([1, 1, 768])
    """

    BACKENDS = {
        "gpt2": "gpt2",
        "gpt2-medium": "gpt2-medium",
        "gpt2-large": "gpt2-large",
        "bert": "bert-base-uncased",
        "distilbert": "distilbert-base-uncased",
    }

    def __init__(
        self,
        backend: str = "gpt2",
        frozen: bool = True,
        output_mode: Literal["last", "mean", "all"] = "mean",
        max_length: int = 77,
        proj_dim: Optional[int] = None,
    ):
        super().__init__()
        self.output_mode = output_mode
        self.max_length = max_length

        model_name = self.BACKENDS.get(backend, backend)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # Handle missing pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.embed_dim = self.model.config.hidden_size

        if frozen:
            for param in self.model.parameters():
                param.requires_grad = False

        self.proj = None
        if proj_dim:
            self.proj = nn.Linear(self.embed_dim, proj_dim)
            self.embed_dim = proj_dim

    def forward(
        self,
        texts: Optional[List[str]] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode text instructions.

        Args:
            texts: List of instruction strings [B]
            input_ids: Pre-tokenized input IDs [B, L]
            attention_mask: Attention mask [B, L]

        Returns:
            Embeddings with shape:
                - [B, 1, D] if output_mode is "last" or "mean"
                - [B, L, D] if output_mode is "all"

        Raises:
            ValueError: If neither texts nor input_ids provided
        """
        if texts is not None:
            device = next(self.model.parameters()).device
            tokens = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            input_ids = tokens["input_ids"].to(device)
            attention_mask = tokens["attention_mask"].to(device)

        if input_ids is None or attention_mask is None:
            raise ValueError("Either texts or input_ids/attention_mask must be provided")

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden: torch.Tensor = outputs.last_hidden_state  # type: ignore

        emb: torch.Tensor
        if self.output_mode == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            emb = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            emb = emb.unsqueeze(1)
        elif self.output_mode == "last":
            seq_len = attention_mask.sum(1) - 1
            emb = hidden[torch.arange(len(hidden)), seq_len].unsqueeze(1)
        else:
            emb = hidden

        return self.proj(emb) if self.proj else emb
