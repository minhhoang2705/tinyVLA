# Phase 05: Language Backbone

## Context Links
- [Tech Stack](../../docs/tech-stack.md) - GPT-2 section
- [VLA Architectures](../reports/researcher-260118-vla-architectures.md) - Language models

## Overview
| Field | Value |
|-------|-------|
| Priority | P1 - Core Component |
| Status | Completed ✓ |
| Effort | 3h |
| Dependencies | Phases 2, 3 |
| Branch | feat/phase-05-language-backbone |
| Commit | f703e9a |

Implement language encoder wrapper for instruction encoding. GPT-2 provides lightweight baseline; architecture supports drop-in replacement with larger LMs.

## Key Insights
- GPT-2 (124M-355M) sufficient for instruction encoding
- Use last hidden state or mean pooling for sentence embedding
- Tokenizer + model must use same vocab
- Freeze language model for stability (RT-2/OpenVLA pattern)

## Requirements

### Functional
- FR-01: Load GPT-2 variants (small, medium, large)
- FR-02: Tokenize text instructions with padding/truncation
- FR-03: Extract embeddings (last token, mean pool, or all)
- FR-04: Optional model freezing
- FR-05: Support batch processing of variable-length instructions

### Non-Functional
- NFR-01: <50ms encoding for 32-token instruction
- NFR-02: Memory <2GB for GPT-2 base

## Architecture

```
src/vla/backbones/
├── __init__.py          # Add language exports
└── language.py          # Language backbone
```

**Language Pipeline:**
```
Text ["pick up block"]
    ↓
Tokenizer (GPT-2)
    ↓
Token IDs [B, L]
    ↓
GPT-2 Model
    ↓
Embeddings [B, L, D] or [B, D]
```

## Related Code Files

### Files to Create
| Path | Purpose | Lines |
|------|---------|-------|
| `src/vla/backbones/language.py` | Language wrapper | ~100 |
| `tests/unit/test_language.py` | Language tests | ~70 |

### Files to Modify
| Path | Changes |
|------|---------|
| `src/vla/backbones/__init__.py` | Add language exports |

## Implementation Steps

### Step 1: Implement language.py (75 min)
```python
"""Language backbone for instruction encoding."""
import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Tokenizer, AutoModel, AutoTokenizer
from typing import Optional, Literal, List, Union

from vla.registry import LANGUAGE_REGISTRY


@LANGUAGE_REGISTRY.register("gpt2")
class GPT2Backbone(nn.Module):
    """GPT-2 language model for instruction encoding.

    Args:
        model_name: HuggingFace model name
        frozen: Freeze model parameters
        output_mode: "last" (last token), "mean" (mean pool), "all" (all tokens)
        max_length: Maximum sequence length
        proj_dim: Optional projection dimension
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

    def _freeze_model(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def tokenize(
        self,
        texts: List[str],
        device: Optional[torch.device] = None,
    ) -> dict:
        """Tokenize batch of texts."""
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        if device:
            tokens = {k: v.to(device) for k, v in tokens.items()}
        return tokens

    def forward(
        self,
        texts: Optional[List[str]] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode text instructions.

        Args:
            texts: List of instruction strings
            input_ids: Pre-tokenized input IDs [B, L]
            attention_mask: Attention mask [B, L]
        Returns:
            Embeddings [B, D] or [B, L, D] depending on output_mode
        """
        # Tokenize if needed
        if texts is not None:
            device = next(self.model.parameters()).device
            tokens = self.tokenize(texts, device)
            input_ids = tokens["input_ids"]
            attention_mask = tokens["attention_mask"]

        # Forward through model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state  # [B, L, D]

        # Extract embeddings based on mode
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

    Provides unified interface for different language models.
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

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state

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
```

### Step 2: Update __init__.py (5 min)
```python
"""Vision and language backbone modules."""
from .vision import VisionBackbone, DINOv2Backbone, SigLIPBackbone
from .feature_extractor import MultiScaleFeatureExtractor, DualEncoderVision
from .language import GPT2Backbone, LanguageEncoder

__all__ = [
    # Vision
    "VisionBackbone",
    "DINOv2Backbone",
    "SigLIPBackbone",
    "MultiScaleFeatureExtractor",
    "DualEncoderVision",
    # Language
    "GPT2Backbone",
    "LanguageEncoder",
]
```

### Step 3: Write tests (40 min)
```python
"""Tests for language backbones."""
import pytest
import torch
from vla.backbones import GPT2Backbone, LanguageEncoder
from vla.registry import LANGUAGE_REGISTRY


class TestGPT2Backbone:
    @pytest.fixture
    def texts(self):
        return ["pick up the red block", "place it on the table"]

    def test_mean_pooling(self, texts):
        encoder = GPT2Backbone(model_name="gpt2", output_mode="mean")
        out = encoder(texts=texts)
        assert out.shape == (2, 1, encoder.embed_dim)

    def test_last_token(self, texts):
        encoder = GPT2Backbone(model_name="gpt2", output_mode="last")
        out = encoder(texts=texts)
        assert out.shape == (2, 1, encoder.embed_dim)

    def test_all_tokens(self, texts):
        encoder = GPT2Backbone(model_name="gpt2", output_mode="all")
        out = encoder(texts=texts)
        assert out.ndim == 3
        assert out.shape[0] == 2
        assert out.shape[2] == encoder.embed_dim

    def test_projection(self, texts):
        encoder = GPT2Backbone(model_name="gpt2", proj_dim=512)
        out = encoder(texts=texts)
        assert out.shape[-1] == 512

    def test_pretokenized_input(self, texts):
        encoder = GPT2Backbone(model_name="gpt2")
        tokens = encoder.tokenize(texts)
        out = encoder(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
        )
        assert out.shape == (2, 1, encoder.embed_dim)

    def test_frozen_params(self, texts):
        encoder = GPT2Backbone(model_name="gpt2", frozen=True)
        for name, param in encoder.model.named_parameters():
            assert not param.requires_grad, f"{name} should be frozen"


class TestLanguageEncoder:
    def test_gpt2_backend(self):
        encoder = LanguageEncoder(backend="gpt2")
        out = encoder(texts=["test instruction"])
        assert out.shape == (1, 1, encoder.embed_dim)

    def test_registry(self):
        assert "gpt2" in LANGUAGE_REGISTRY
        assert "language_encoder" in LANGUAGE_REGISTRY
```

## Todo List
- [x] Implement GPT2Backbone with output modes
- [x] Implement LanguageEncoder generic wrapper
- [x] Handle tokenizer pad token edge cases
- [x] Register in LANGUAGE_REGISTRY
- [x] Update backbones __init__.py
- [x] Write unit tests (23 tests, all passing)
- [x] Test with real pretrained weights
- [x] Verify frozen parameters

**Implementation Complete:** All tasks finished, all tests pass, type checking clean.

## Success Criteria
1. GPT2Backbone encodes text to embeddings
2. All output modes (last/mean/all) work correctly
3. Variable-length batch processing works
4. Freezing prevents gradient flow
5. All tests pass

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Tokenizer mismatch | High | Always load from same model_name |
| OOM with long sequences | Medium | Default max_length=77, truncation |
| Pad token issues | Medium | Explicit pad_token assignment |

## Security Considerations
- Models loaded from HuggingFace Hub (trusted)
- No user-generated code execution

## Next Steps
- Phase 6: Fusion mechanisms to combine vision + language
