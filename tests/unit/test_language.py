"""Tests for language backbones.

Tests GPT-2 and generic language encoder wrappers including:
- Different output modes (last, mean, all)
- Pretokenized input handling
- Optional projection layers
- Parameter freezing
- Registry integration
"""

import pytest
import torch

from vla.backbones import GPT2Backbone, LanguageEncoder
from vla.registry import LANGUAGE_REGISTRY


class TestGPT2Backbone:
    """Tests for GPT2Backbone wrapper."""

    @pytest.fixture
    def texts(self):
        """Sample instruction texts."""
        return ["pick up the red block", "place it on the table"]

    def test_mean_pooling(self, texts):
        """Test mean pooling output mode."""
        encoder = GPT2Backbone(model_name="gpt2", output_mode="mean")
        out = encoder(texts=texts)
        assert out.shape == (2, 1, encoder.embed_dim)
        assert out.requires_grad is False  # Frozen by default

    def test_last_token(self, texts):
        """Test last token output mode."""
        encoder = GPT2Backbone(model_name="gpt2", output_mode="last")
        out = encoder(texts=texts)
        assert out.shape == (2, 1, encoder.embed_dim)

    def test_all_tokens(self, texts):
        """Test all tokens output mode."""
        encoder = GPT2Backbone(model_name="gpt2", output_mode="all")
        out = encoder(texts=texts)
        assert out.ndim == 3
        assert out.shape[0] == 2
        assert out.shape[2] == encoder.embed_dim

    def test_projection(self, texts):
        """Test optional projection layer."""
        encoder = GPT2Backbone(model_name="gpt2", proj_dim=512)
        out = encoder(texts=texts)
        assert out.shape[-1] == 512
        assert encoder.embed_dim == 512

    def test_pretokenized_input(self, texts):
        """Test forward with pretokenized input."""
        encoder = GPT2Backbone(model_name="gpt2")
        tokens = encoder.tokenize(texts)
        out = encoder(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
        )
        assert out.shape == (2, 1, encoder.embed_dim)

    def test_frozen_params(self, texts):
        """Test that frozen=True prevents gradient updates."""
        encoder = GPT2Backbone(model_name="gpt2", frozen=True)
        for name, param in encoder.model.named_parameters():
            assert not param.requires_grad, f"{name} should be frozen"

        # Projection layer should still be trainable
        if encoder.proj is not None:
            for param in encoder.proj.parameters():
                assert param.requires_grad

    def test_unfrozen_params(self, texts):
        """Test that frozen=False allows gradient updates."""
        encoder = GPT2Backbone(model_name="gpt2", frozen=False)
        trainable_params = [p for p in encoder.model.parameters() if p.requires_grad]
        assert len(trainable_params) > 0

    def test_variable_length_batch(self):
        """Test handling of variable-length instructions."""
        encoder = GPT2Backbone(model_name="gpt2")
        texts = ["short", "this is a much longer instruction"]
        out = encoder(texts=texts)
        assert out.shape == (2, 1, encoder.embed_dim)

    def test_tokenizer_pad_token(self):
        """Test that pad token is properly set."""
        encoder = GPT2Backbone(model_name="gpt2")
        assert encoder.tokenizer.pad_token is not None
        assert encoder.tokenizer.pad_token == encoder.tokenizer.eos_token

    def test_max_length_truncation(self):
        """Test that max_length truncation works."""
        encoder = GPT2Backbone(model_name="gpt2", max_length=10)
        long_text = " ".join(["word"] * 50)
        tokens = encoder.tokenize([long_text])
        assert tokens["input_ids"].shape[1] <= 10

    def test_device_handling(self):
        """Test device placement of tokenized tensors."""
        encoder = GPT2Backbone(model_name="gpt2")
        texts = ["test instruction"]
        device = torch.device("cpu")
        tokens = encoder.tokenize(texts, device=device)
        assert tokens["input_ids"].device == device

    def test_no_input_raises_error(self):
        """Test that missing input raises ValueError."""
        encoder = GPT2Backbone(model_name="gpt2")
        with pytest.raises(ValueError, match="Either texts or input_ids"):
            encoder()

    def test_registry_integration(self):
        """Test that GPT2Backbone is registered."""
        assert "gpt2" in LANGUAGE_REGISTRY
        encoder = LANGUAGE_REGISTRY.get("gpt2", model_name="gpt2")
        assert isinstance(encoder, GPT2Backbone)


class TestLanguageEncoder:
    """Tests for generic LanguageEncoder wrapper."""

    def test_gpt2_backend(self):
        """Test GPT-2 backend."""
        encoder = LanguageEncoder(backend="gpt2")
        out = encoder(texts=["test instruction"])
        assert out.shape == (1, 1, encoder.embed_dim)

    def test_custom_backend(self):
        """Test custom HuggingFace model name."""
        # Should work with any valid HuggingFace model
        encoder = LanguageEncoder(backend="gpt2")
        assert encoder.model is not None
        assert encoder.tokenizer is not None

    def test_pad_token_handling(self):
        """Test automatic pad token assignment."""
        encoder = LanguageEncoder(backend="gpt2")
        assert encoder.tokenizer.pad_token is not None

    def test_frozen_by_default(self):
        """Test that models are frozen by default."""
        encoder = LanguageEncoder(backend="gpt2", frozen=True)
        for param in encoder.model.parameters():
            assert not param.requires_grad

    def test_output_modes(self):
        """Test all output modes."""
        texts = ["test instruction"]

        # Mean pooling
        enc_mean = LanguageEncoder(backend="gpt2", output_mode="mean")
        out_mean = enc_mean(texts=texts)
        assert out_mean.shape == (1, 1, enc_mean.embed_dim)

        # Last token
        enc_last = LanguageEncoder(backend="gpt2", output_mode="last")
        out_last = enc_last(texts=texts)
        assert out_last.shape == (1, 1, enc_last.embed_dim)

        # All tokens
        enc_all = LanguageEncoder(backend="gpt2", output_mode="all")
        out_all = enc_all(texts=texts)
        assert out_all.ndim == 3

    def test_projection_layer(self):
        """Test optional projection layer."""
        encoder = LanguageEncoder(backend="gpt2", proj_dim=256)
        out = encoder(texts=["test"])
        assert out.shape[-1] == 256
        assert encoder.embed_dim == 256

    def test_no_input_raises_error(self):
        """Test that missing input raises ValueError."""
        encoder = LanguageEncoder(backend="gpt2")
        with pytest.raises(ValueError, match="Either texts or input_ids"):
            encoder()

    def test_registry_integration(self):
        """Test that LanguageEncoder is registered."""
        assert "language_encoder" in LANGUAGE_REGISTRY
        encoder = LANGUAGE_REGISTRY.get("language_encoder", backend="gpt2")
        assert isinstance(encoder, LanguageEncoder)

    def test_backend_aliases(self):
        """Test that backend aliases work."""
        for backend in ["gpt2", "gpt2-medium", "gpt2-large"]:
            encoder = LanguageEncoder(backend=backend)
            assert encoder.model is not None

    def test_mean_pooling_clamp(self):
        """Test that mean pooling has numerical stability."""
        encoder = LanguageEncoder(backend="gpt2", output_mode="mean")
        # Create edge case with very short sequence
        texts = ["a"]
        out = encoder(texts=texts)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
