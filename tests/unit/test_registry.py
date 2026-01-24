"""Unit tests for registry module."""

import pytest
import torch.nn as nn
from omegaconf import DictConfig

from vla.registry import (
    ACTION_REGISTRY,
    FUSION_REGISTRY,
    LANGUAGE_REGISTRY,
    MODEL_REGISTRY,
    VISION_REGISTRY,
    Registry,
    build_action_head,
    build_fusion_module,
    build_language_encoder,
    build_model,
    build_vision_encoder,
)


class TestRegistry:
    """Test suite for Registry class."""

    def test_register_and_get(self):
        """Test basic registration and instantiation."""
        registry = Registry[nn.Module]("test")

        @registry.register("dummy")
        class DummyModule(nn.Module):
            def __init__(self, dim: int = 64):
                super().__init__()
                self.dim = dim

        module = registry.get("dummy", dim=128)
        assert isinstance(module, DummyModule)
        assert module.dim == 128

    def test_register_with_default_args(self):
        """Test registration with default constructor arguments."""
        registry = Registry[nn.Module]("test")

        @registry.register("with_defaults")
        class ModuleWithDefaults(nn.Module):
            def __init__(self, dim: int = 256, num_layers: int = 4):
                super().__init__()
                self.dim = dim
                self.num_layers = num_layers

        # Test with defaults
        module1 = registry.get("with_defaults")
        assert module1.dim == 256
        assert module1.num_layers == 4

        # Test with overrides
        module2 = registry.get("with_defaults", dim=512, num_layers=8)
        assert module2.dim == 512
        assert module2.num_layers == 8

    def test_duplicate_registration_raises(self):
        """Test that duplicate registration raises ValueError."""
        registry = Registry[nn.Module]("test")

        @registry.register("same_name")
        class First(nn.Module):
            pass

        with pytest.raises(ValueError, match="already registered"):

            @registry.register("same_name")
            class Second(nn.Module):
                pass

    def test_unknown_component_raises(self):
        """Test that unknown component raises KeyError with helpful message."""
        registry = Registry[nn.Module]("test")

        with pytest.raises(KeyError, match="not found"):
            registry.get("nonexistent")

    def test_unknown_component_lists_available(self):
        """Test that error message lists available components."""
        registry = Registry[nn.Module]("test")

        @registry.register("component_a")
        class ComponentA(nn.Module):
            pass

        @registry.register("component_b")
        class ComponentB(nn.Module):
            pass

        with pytest.raises(KeyError, match="component_a, component_b"):
            registry.get("missing")

    def test_list_available(self):
        """Test listing all registered components."""
        registry = Registry[nn.Module]("test")

        @registry.register("a")
        class A(nn.Module):
            pass

        @registry.register("b")
        class B(nn.Module):
            pass

        @registry.register("c")
        class C(nn.Module):
            pass

        available = registry.list_available()
        assert available == ["a", "b", "c"]  # Should be sorted

    def test_list_available_empty_registry(self):
        """Test listing components in empty registry."""
        registry = Registry[nn.Module]("empty")
        assert registry.list_available() == []

    def test_contains(self):
        """Test __contains__ operator."""
        registry = Registry[nn.Module]("test")

        @registry.register("exists")
        class Exists(nn.Module):
            pass

        assert "exists" in registry
        assert "missing" not in registry

    def test_get_class(self):
        """Test retrieving class without instantiation."""
        registry = Registry[nn.Module]("test")

        @registry.register("my_module")
        class MyModule(nn.Module):
            def __init__(self, dim: int):
                super().__init__()
                self.dim = dim

        cls = registry.get_class("my_module")
        assert cls is MyModule

        # Can instantiate manually
        instance = cls(dim=512)
        assert instance.dim == 512

    def test_get_class_raises_for_unknown(self):
        """Test get_class raises KeyError for unknown component."""
        registry = Registry[nn.Module]("test")

        with pytest.raises(KeyError, match="not found"):
            registry.get_class("unknown")

    def test_repr(self):
        """Test string representation of registry."""
        registry = Registry[nn.Module]("test_registry")

        @registry.register("component")
        class Component(nn.Module):
            pass

        repr_str = repr(registry)
        assert "test_registry" in repr_str
        assert "components=1" in repr_str


class TestGlobalRegistries:
    """Test suite for global registry instances."""

    def test_global_registries_exist(self):
        """Test that all global registries are initialized."""
        assert VISION_REGISTRY is not None
        assert LANGUAGE_REGISTRY is not None
        assert FUSION_REGISTRY is not None
        assert ACTION_REGISTRY is not None
        assert MODEL_REGISTRY is not None

    def test_global_registries_names(self):
        """Test that global registries have correct names."""
        assert VISION_REGISTRY._name == "vision"
        assert LANGUAGE_REGISTRY._name == "language"
        assert FUSION_REGISTRY._name == "fusion"
        assert ACTION_REGISTRY._name == "action"
        assert MODEL_REGISTRY._name == "model"

    def test_global_registries_independent(self):
        """Test that global registries are independent."""

        @VISION_REGISTRY.register("shared_name")
        class VisionComponent(nn.Module):
            pass

        @LANGUAGE_REGISTRY.register("shared_name")
        class LanguageComponent(nn.Module):
            pass

        # Should not raise - different registries
        vision_comp = VISION_REGISTRY.get("shared_name")
        language_comp = LANGUAGE_REGISTRY.get("shared_name")

        assert isinstance(vision_comp, VisionComponent)
        assert isinstance(language_comp, LanguageComponent)


class TestFactoryFunctions:
    """Test suite for factory functions."""

    def test_build_vision_encoder_from_registry(self):
        """Test building vision encoder using registry."""

        @VISION_REGISTRY.register("test_vision")
        class TestVisionEncoder(nn.Module):
            def __init__(self, dim: int = 512):
                super().__init__()
                self.dim = dim

        cfg = DictConfig({"name": "test_vision", "dim": 768})
        encoder = build_vision_encoder(cfg)

        assert isinstance(encoder, TestVisionEncoder)
        assert encoder.dim == 768

    def test_build_language_encoder_from_registry(self):
        """Test building language encoder using registry."""

        @LANGUAGE_REGISTRY.register("test_language")
        class TestLanguageEncoder(nn.Module):
            def __init__(self, vocab_size: int = 10000):
                super().__init__()
                self.vocab_size = vocab_size

        cfg = DictConfig({"name": "test_language", "vocab_size": 50000})
        encoder = build_language_encoder(cfg)

        assert isinstance(encoder, TestLanguageEncoder)
        assert encoder.vocab_size == 50000

    def test_build_fusion_module_from_registry(self):
        """Test building fusion module using registry."""

        @FUSION_REGISTRY.register("test_fusion")
        class TestFusion(nn.Module):
            def __init__(self, num_latents: int = 64):
                super().__init__()
                self.num_latents = num_latents

        cfg = DictConfig({"name": "test_fusion", "num_latents": 128})
        fusion = build_fusion_module(cfg)

        assert isinstance(fusion, TestFusion)
        assert fusion.num_latents == 128

    def test_build_action_head_from_registry(self):
        """Test building action head using registry."""

        @ACTION_REGISTRY.register("test_action")
        class TestActionHead(nn.Module):
            def __init__(self, action_dim: int = 7):
                super().__init__()
                self.action_dim = action_dim

        cfg = DictConfig({"name": "test_action", "action_dim": 14})
        head = build_action_head(cfg)

        assert isinstance(head, TestActionHead)
        assert head.action_dim == 14

    def test_build_model_from_registry(self):
        """Test building model using registry."""

        @MODEL_REGISTRY.register("test_model")
        class TestModel(nn.Module):
            def __init__(self, hidden_dim: int = 512):
                super().__init__()
                self.hidden_dim = hidden_dim

        cfg = DictConfig({"name": "test_model", "hidden_dim": 1024})
        model = build_model(cfg)

        assert isinstance(model, TestModel)
        assert model.hidden_dim == 1024

    def test_build_with_hydra_target(self):
        """Test factory functions with Hydra _target_ pattern."""

        # Create a simple test class
        class HydraTestModule(nn.Module):
            def __init__(self, value: int):
                super().__init__()
                self.value = value

        # Register in global scope for Hydra instantiate
        import sys

        sys.modules[__name__].HydraTestModule = HydraTestModule

        cfg = DictConfig({"_target_": "tests.unit.test_registry.HydraTestModule", "value": 999})

        module = build_vision_encoder(cfg)
        assert isinstance(module, HydraTestModule)
        assert module.value == 999
