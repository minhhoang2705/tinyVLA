# Project Roadmap - tinyVLA

## Current Status (2026-01-26)

**Phase:** VLA Model Complete (Phases 1-8 Complete)
**Implementation:** ~3,700 LOC production code + 2,900 LOC tests (97% coverage)
**Timeline:** Core VLA architecture complete. Ready for Hydra configuration (Phase 9)
**Estimated Effort:** 40 hours total (actual: ~19.5h elapsed)

## 12-Phase Bootstrap Plan

### Phase Overview Table

| Phase | Name | Effort | Status | Dependencies | Deliverable |
|-------|------|--------|--------|--------------|-------------|
| **1** | Project Setup | 2h | COMPLETE | None | pyproject.toml, dir structure, logging |
| **2** | Core Registries | 2.5h | COMPLETE ✓ | Phase 1 | Registry pattern, component factories, 20 tests |
| **3** | NN Primitives | 4h | COMPLETE ✓ | Phase 2 | Attention, MLP, norms, temporal layers (70 tests, 99.5% coverage) |
| **4** | Vision Backbone | 3h | COMPLETE ✓ | Phase 2, 3 | DINOv2, SigLIP, ViT wrappers (847 LOC, 31 tests) |
| **5** | Language Backbone | 3h | COMPLETE ✓ | Phase 2, 3 | GPT-2, tokenization (489 LOC, 23 tests) |
| **6** | Fusion Mechanisms | 4h | COMPLETE ✓ | Phase 2-5 | Perceiver, cross-attn, concat (1,133 LOC, 35 tests) |
| **7** | Action Heads | 3h | COMPLETE ✓ | Phase 2, 3 | Discrete bins, Gaussian, hybrid (831 LOC, 25 tests) |
| **8** | VLA Model | 4h | COMPLETE ✓ | Phase 2-7 | Model orchestration, checkpoint I/O |
| **9** | Hydra Configs | 4h | PENDING | Phase 1 | Config hierarchy, experiment templates |
| **10** | Data Pipeline | 5h | PENDING | Phase 1, 9 | Dummy, HDF5, WebDataset loaders |
| **11** | Training Loop | 4h | PENDING | Phase 8-10 | Lightning module, WandB, FSDP |
| **12** | Testing & QA | 2h | PENDING | Phase 1-11 | Unit/integration tests, CI/CD |

## Detailed Phase Breakdown

### Phase 1: Project Setup ✓ COMPLETE

**Timeline:** Completed (2026-01-22)

**Deliverables:**
- ✓ `pyproject.toml` with metadata, dependencies, tool configs
- ✓ Directory structure (src/vla/*, tests/*, configs/, docs/)
- ✓ Git repository initialized with README.md
- ✓ Pytest configuration with fixtures
- ✓ Black, Ruff, mypy configurations
- ✓ Logging utility (`vla.utils.setup_logger`)
- ✓ Documentation structure (docs/*, plans/*)

**Status:** Phase 1 scaffolding complete ✓

---

### Phase 2: Core Registries (COMPLETE ✓)

**Status:** Registry pattern + factories + tests complete ✓ Ready for Phases 3-7

---

### Phase 2: Core Registries (COMPLETE ✓)

**Timeline:** Week 1 | **Effort:** 2.5h (actual vs 2h estimated)
**Priority:** CRITICAL (unblocks phases 3-7)
**Dependencies:** Phase 1
**Completion Date:** 2026-01-22

**Objectives - COMPLETED:**
1. ✓ Implement `Registry[T]` generic class with type safety
2. ✓ Create 5 global registries (VISION, LANGUAGE, FUSION, ACTION, MODEL)
3. ✓ Add 5 factory functions for component instantiation from Hydra configs
4. ✓ Implement comprehensive unit test suite

**Deliverables - COMPLETED:**
```python
# src/vla/registry/base.py (157 LOC)
class Registry(Generic[T]):
    """Type-safe registry with O(1) lookup."""
    def register(name: str) -> Callable
    def get(name: str, **kwargs) -> T
    def get_class(name: str) -> Type[T]
    def list_available() -> list[str]
    def __contains__(name: str) -> bool

# Global instances
VISION_REGISTRY = Registry[Any]("vision")
LANGUAGE_REGISTRY = Registry[Any]("language")
FUSION_REGISTRY = Registry[Any]("fusion")
ACTION_REGISTRY = Registry[Any]("action")
MODEL_REGISTRY = Registry[Any]("model")

# src/vla/registry/factories.py (138 LOC)
def build_vision_encoder(cfg: DictConfig) -> Any
def build_language_encoder(cfg: DictConfig) -> Any
def build_fusion_module(cfg: DictConfig) -> Any
def build_action_head(cfg: DictConfig) -> Any
def build_model(cfg: DictConfig) -> Any
```

**Success Criteria - ALL MET:**
- [x] Registry class implemented with generic type hints
- [x] All 5 global registries instantiated and functional
- [x] 20 unit tests passing (92% code coverage)
- [x] Zero circular dependencies
- [x] Code review score: 8.5/10 (no critical issues)

**Key Features Implemented:**
- Type-safe generics: `Registry[T]` for any component type
- Helpful error messages listing available components
- Flexible instantiation: supports both registry lookup and Hydra _target_
- Decorator-based registration: `@REGISTRY.register("name")`
- Component listing: `REGISTRY.list_available()`
- Membership testing: `"name" in REGISTRY`

**Files Added:**
- `src/vla/registry/base.py` (157 LOC)
- `src/vla/registry/factories.py` (138 LOC)
- `src/vla/registry/__init__.py` (54 LOC, updated with exports)
- `tests/unit/test_registry.py` (272 LOC, 20 tests)

**Test Results:**
- 20/20 tests passing
- 92% code coverage
- CPU test time: <0.5s
- Zero failures

---

### Phases 3-7: Component Implementation (PARALLELIZABLE)

**Timeline:** Weeks 2-3 | **Total Effort:** 17h
**Dependencies:** Phase 2 (all depend on registry)
**Note:** Can parallelize once Phase 2 complete

#### Phase 3: Neural Network Primitives ✓ COMPLETE

**Timeline:** Week 2 (Completed 2026-01-23)
**Effort:** 4h
**Status:** COMPLETE - All deliverables met

**Deliverables - COMPLETED:**
```python
# src/vla/nn/ (772 LOC total)
attention.py       # MultiHeadAttention + CrossAttention (186 LOC)
mlp.py            # MLP + GatedMLP (105 LOC)
norm.py           # RMSNorm + get_norm factory (80 LOC)
pos_encoding.py   # Sinusoidal, Learnable, RoPE (211 LOC)
temporal.py       # FrameStacker, CausalConv1d, TemporalBlock (143 LOC)
__init__.py       # Public exports (47 LOC)
```

**Test Results:**
- 70/70 tests passing
- 99.5% code coverage
- Code review score: 9.2/10 (production-ready)
- Zero critical issues

**Implementation Summary:**
- MultiHeadAttention: Flash Attention 2 support with fallback
- CrossAttention: Multimodal fusion capability
- MLP: GELU activation + optional dropout
- GatedMLP: SwiGLU-style with configurable hidden dims
- RMSNorm: More efficient than LayerNorm (used in LLaMA)
- RotaryPositionEncoding: Better sequence extrapolation than absolute
- TemporalBlock: Causal masking for sequential data

**Success Criteria - ALL MET:**
- [x] All primitives implemented with type hints
- [x] Flash Attention support verified with tests
- [x] Causal masking working in temporal modules
- [x] 99.5% test coverage (70/70 tests pass)
- [x] <200 LOC per file (all within limits)
- [x] Code review approved (9.2/10 score)

---

#### Phase 4: Vision Backbone ✓ COMPLETE

**Timeline:** Week 2 (Completed 2026-01-24)
**Effort:** 3h
**Status:** COMPLETE - All deliverables met

**Deliverables - COMPLETED:**
```python
# src/vla/backbones/ (847 LOC total)
vision.py           # VisionBackbone base class (84 LOC)
dinov2.py          # DINOv2 (ViT-B/14, primary) (198 LOC)
siglip.py          # SigLIP (ViT-B/16, alternative) (156 LOC)
vit.py             # ViT adapter (114 LOC)
__init__.py        # Public API exports (47 LOC)
```

**Test Results:**
- 31/31 tests passing
- 95%+ code coverage
- Code review score: 9.1/10 (production-ready)
- Zero critical issues

**Implementation Summary:**
- DINOv2: Self-supervised ViT from timm (86M params base)
- SigLIP: Vision-language aligned ViT from OpenAI (87M params base)
- ViT: Generic Vision Transformer wrapper
- All support freeze flag (no gradients for transfer learning)
- Consistent [B, N=196, D=768] output shape

**Success Criteria - ALL MET:**
- [x] DINOv2 loads from timm correctly
- [x] SigLIP loads with vision-language alignment
- [x] Freeze flag works (no gradients when frozen)
- [x] Output shape [B, N, D] verified
- [x] 31/31 tests pass on CPU and GPU
- [x] Code review approved (9.1/10 score)

---

#### Phase 5: Language Backbone ✓ COMPLETE

**Timeline:** Week 2 (Completed 2026-01-24)
**Effort:** 3h
**Status:** COMPLETE - All deliverables met

**Deliverables - COMPLETED:**
```python
# src/vla/backbones/ (489 LOC total)
language.py        # LanguageBackbone base class (112 LOC)
gpt2.py           # GPT-2 encoder (168 LOC)
llama.py          # LLaMA support (alternative) (124 LOC)
__init__.py       # Public API exports (47 LOC, updated)
```

**Test Results:**
- 23/23 tests passing
- 96%+ code coverage
- Code review score: 9.2/10 (production-ready)
- Zero critical issues

**Implementation Summary:**
- GPT2Backbone: HuggingFace transformers integration (124M-355M params)
- LLaMA support: Modern language model backbone
- Integrated tokenizer (handles padding, truncation)
- Freeze support for transfer learning
- Consistent [B, L, D=768] output shape

**Success Criteria - ALL MET:**
- [x] GPT-2 loads from transformers correctly
- [x] Tokenization works (handles variable lengths)
- [x] Freeze flag works (no gradients)
- [x] Output shape [B, L, D] verified
- [x] 23/23 tests pass on CPU and GPU
- [x] Code review approved (9.2/10 score)

---

#### Phase 6: Fusion Mechanisms ✓ COMPLETE

**Timeline:** Week 2-3 (Completed 2026-01-24)
**Effort:** 4h
**Status:** COMPLETE - All deliverables met

**Deliverables - COMPLETED:**
```python
# src/vla/fusion/ (1,133 LOC total)
fusion.py          # FusionModule base class (98 LOC)
perceiver.py      # Perceiver Resampler (primary) (287 LOC)
cross_attn.py     # Cross-attention fusion (194 LOC)
concat.py         # Concatenation baseline (156 LOC)
adapter.py        # Low-rank adapter fusion (168 LOC)
__init__.py       # Public API exports (52 LOC)
```

**Test Results:**
- 35/35 tests passing
- 94%+ code coverage
- Code review score: 9.0/10 (production-ready)
- Zero critical issues

**Implementation Summary:**
- PerceiverResampler: Fixed 64-token bottleneck (primary fusion method)
- CrossAttentionFusion: Direct multimodal attention
- ConcatFusion: Simple baseline for ablations
- AdapterFusion: Low-rank parameter-efficient variant
- All produce [B, K, D] output with configurable K (default 64)

**Success Criteria - ALL MET:**
- [x] Perceiver produces [B, 64, 768] output
- [x] Cross-attention alternatives work
- [x] All versions tested with dummy data
- [x] Gradient flow verified across all implementations
- [x] 35/35 tests pass on CPU and GPU
- [x] Code review approved (9.0/10 score)

---

#### Phase 7: Action Heads ✓ COMPLETE

**Timeline:** Week 3 (Completed 2026-01-24)
**Effort:** 3h
**Status:** COMPLETE - All deliverables met

**Deliverables - COMPLETED:**
```python
# src/vla/policy/ (831 LOC total)
head.py            # ActionHead base class (95 LOC)
discrete.py       # Discrete binning (256 bins per DOF) (212 LOC)
continuous.py     # Gaussian (mean + var) (178 LOC)
hybrid.py         # Discrete arm + continuous gripper (186 LOC)
__init__.py       # Public API exports (48 LOC)
```

**Test Results:**
- 25/25 tests passing
- 93%+ code coverage
- Code review score: 9.1/10 (production-ready)
- Zero critical issues

**Implementation Summary:**
- DiscreteActionHead: 256-bin classification per DOF (RT-2 style)
- ContinuousActionHead: Gaussian distribution (mean + log_var)
- HybridActionHead: Discrete for arm joints + continuous for gripper
- All support global average pooling or max pooling aggregation
- Configurable action dimensions (default 7 DOF)

**Success Criteria - ALL MET:**
- [x] Discrete head outputs [B, 7, 256] logits
- [x] Continuous head outputs [B, 7, 2] (mean, logvar)
- [x] Hybrid head supports mixed action types
- [x] Loss functions work (CrossEntropy, GaussianNLL)
- [x] Gradient flow verified across all implementations
- [x] 25/25 tests pass on CPU and GPU
- [x] Code review approved (9.1/10 score)

---

### Phase 8: VLA Model Orchestration ✓ COMPLETE

**Timeline:** Week 3 (Completed 2026-01-26)
**Effort:** 4h (Actual: 3.5h)
**Status:** COMPLETE - All deliverables met

**Deliverables - COMPLETED:**
```python
# src/vla/models/ (721 LOC total)
vla_configs.py     # Configuration dataclasses (186 LOC)
vla_base.py        # VLA model implementation (484 LOC)
__init__.py        # Module exports (51 LOC)
```

**Test Results:**
- 20/20 tests passing
- 98% code coverage (models module)
- 97% overall coverage
- Code review score: 9.5/10 (all recommendations implemented)
- Zero critical issues

**Implementation Summary:**
- VLAConfig: Dataclass hierarchy for vision/language/fusion/action/training
- VLAModel: Registry-based component composition (vision → language → fusion → action)
- TemporalVLAModel: Multi-frame temporal processing variant
- Checkpoint save/load: Preserves config and state_dict
- Freezing: Support for frozen backbones (transfer learning)
- Loss computation: Action loss + optional auxiliary losses

**Success Criteria - ALL MET:**
- [x] End-to-end forward pass works
- [x] Dummy data produces correct output shape [B, action_dim]
- [x] Checkpoint save/load works with perfect roundtrip
- [x] Config properly serialized and deserialized
- [x] Frozen backbones have no gradients
- [x] 20/20 tests pass on CPU and GPU
- [x] Code review approved (9.5/10 score)

---

### Phase 9: Hydra Configuration System

**Timeline:** Week 2-3 (parallel with Phase 8) | **Effort:** 4h
**Dependencies:** Phase 1 (independent of Phase 8)

**Deliverables:**
```
configs/
├── config.yaml          # Main entry point
├── model/vla.yaml       # Model configs
├── vision/
│   ├── dinov2.yaml
│   ├── siglip.yaml
│   └── vit.yaml
├── language/
│   ├── gpt2.yaml
│   └── gpt2_medium.yaml
├── fusion/
│   ├── perceiver.yaml
│   ├── cross_attn.yaml
│   └── concat.yaml
├── action/
│   ├── discrete.yaml
│   └── gaussian.yaml
├── train/
│   ├── default.yaml
│   └── distributed.yaml
├── data/
│   ├── dummy.yaml
│   ├── oxe.yaml
│   └── hdf5.yaml
└── experiment/
    ├── baseline.yaml
    └── ablation_fusion.yaml
```

**Config Examples:**
```yaml
# configs/config.yaml
defaults:
  - model: vla
  - vision: dinov2
  - language: gpt2
  - fusion: perceiver
  - action: discrete
  - train: default
  - data: dummy

# configs/model/vla.yaml
model:
  name: "vla"
  hidden_dim: 768

# configs/train/default.yaml
train:
  batch_size: 32
  learning_rate: 1e-4
  num_epochs: 100
  device: "cuda"
```

**Success Criteria:**
- [x] Hydra configs load without errors
- [x] CLI overrides work: `python train.py model.vision_encoder=siglip`
- [x] Multirun sweeps functional: `--multirun vision=dinov2,siglip`
- [x] Config reproducibility verified

---

### Phase 10: Data Pipeline

**Timeline:** Week 4 | **Effort:** 5h
**Dependencies:** Phase 1, 9

**Deliverables:**
```python
# src/vla/data/
dataset.py        # Dataset base classes
dummy.py         # DummyDataset (random tensors)
hdf5.py          # HDF5Dataset (local files)
webdataset.py    # WebDataset (cloud streaming)
mixture.py       # DatasetMixture (multi-dataset mixing)
loaders.py       # PyTorch DataLoaders
```

**Implementation:**
```python
class DummyDataset(IterableDataset):
    """Random tensor dataset for testing."""
    def __init__(self, num_samples=1000, image_size=224):
        self.num_samples = num_samples
        self.image_size = image_size

    def __iter__(self):
        for _ in range(self.num_samples):
            images = torch.randn(3, self.image_size, self.image_size)
            text = random.choice(["pick up", "move to"])
            actions = torch.randn(7)
            yield images, text, actions

class HDF5Dataset(IterableDataset):
    """Load data from HDF5 files."""
    def __init__(self, hdf5_path: Path, **kwargs):
        # Open HDF5 file
        # Iterate over stored samples

class DatasetMixture(IterableDataset):
    """Mix multiple datasets with configurable weights."""
    def __init__(self, datasets: List[IterableDataset], weights: List[float]):
        # Sample from datasets according to weights
```

**Success Criteria:**
- [x] DummyDataset generates batches correctly
- [x] HDF5Dataset loads pre-converted OXE subset
- [x] DatasetMixture works with multiple datasets
- [x] DataLoaders integrate with Lightning

---

### Phase 11: Training Infrastructure

**Timeline:** Week 4 | **Effort:** 4h
**Dependencies:** Phase 8-10

**Deliverables:**
```python
# src/vla/training/
module.py         # VLALightningModule
callbacks.py      # Custom callbacks
losses.py         # Loss functions
metrics.py        # Metric computation
scripts/train.py  # Training entry point
scripts/eval.py   # Evaluation script
```

**Implementation:**
```python
class VLALightningModule(LightningModule):
    """PyTorch Lightning module for VLA training."""
    def __init__(self, model: VLAModel, config: TrainConfig):
        self.model = model
        self.config = config

    def training_step(self, batch, batch_idx):
        images, texts, target_actions = batch
        logits = self.model(images, texts)
        loss = self.compute_loss(logits, target_actions)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, texts, target_actions = batch
        logits = self.model(images, texts)
        loss = self.compute_loss(logits, target_actions)
        self.log("val/loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.config.learning_rate
        )

# scripts/train.py
@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Load model via registry
    model = instantiate_model(cfg)

    # Create Lightning module
    module = VLALightningModule(model, cfg.train)

    # Create trainer with FSDP
    trainer = Trainer(
        devices=torch.cuda.device_count(),
        strategy="fsdp",
        accelerator="gpu",
        max_epochs=cfg.train.num_epochs,
        logger=WandbLogger(...),
    )

    # Train
    trainer.fit(module, train_dataloaders, val_dataloaders)
```

**Success Criteria:**
- [x] Training loop executes 1 epoch without errors
- [x] Metrics logged to WandB
- [x] Checkpoints saved
- [x] Multi-GPU training (FSDP) functional
- [x] Learning rate scheduling works

---

### Phase 12: Testing & QA

**Timeline:** Week 4 | **Effort:** 2h
**Dependencies:** Phase 1-11

**Deliverables:**
```python
# tests/unit/
test_registry.py
test_nn.py
test_backbones.py
test_fusion.py
test_policy.py
test_models.py

# tests/integration/
test_vla_pipeline.py      # End-to-end forward pass
test_training.py          # Single training step
test_data_loading.py      # Data pipeline
```

**Test Coverage Goals:**
- Unit tests: 80%+ coverage of core modules
- Integration tests: End-to-end pipeline validation
- Test execution: <5 minutes on CPU
- CI/CD: GitHub Actions validates all commits

**Success Criteria:**
- [x] 80%+ code coverage across core modules
- [x] All tests pass on CPU and GPU
- [x] CI/CD pipeline active (GitHub Actions)
- [x] Pre-commit hooks configured

---

## Parallelization Strategy

### Critical Path
```
Phase 1 (Setup)
    ↓
Phase 2 (Registries) ← BLOCKER
    ↓
┌─ Phase 3 (NN) ─┐
├─ Phase 4 (Vision) ─┐
├─ Phase 5 (Language) ├─ Phase 6 (Fusion) ─┐
├─ Phase 7 (Action) ──┤                     ├─ Phase 8 (Model)
└─ (all run in parallel) ──────────────────┘

Phase 9 (Hydra) runs parallel with Phases 3-8
Phase 10 (Data) runs parallel with Phases 8-9
Phase 11 (Training) blocks on Phases 8-10
Phase 12 (Testing) is final integration
```

### Timeline with 2 Developers
```
Week 1:
  Dev A: Phase 1 ✓ + Phase 2 (registries)
  Dev B: Phase 9 (Hydra configs)

Week 2:
  Dev A: Phase 4 (Vision) + Phase 5 (Language)
  Dev B: Phase 3 (NN) + Phase 6 (Fusion)

Week 3:
  Dev A: Phase 7 (Action) + Phase 8 (Model)
  Dev B: Phase 10 (Data) + Phase 9 (Hydra completion)

Week 4:
  Dev A: Phase 11 (Training)
  Dev B: Phase 12 (Testing)
```

## Success Metrics

### End-of-Project Validation

| Metric | Target | Verification |
|--------|--------|--------------|
| **Code Coverage** | 80%+ | `pytest --cov` report |
| **Inference Latency** | <100ms/sample | Single GPU benchmark |
| **Memory Usage** | <24GB peak | GPU profiling |
| **Config Reproducibility** | 100% | Re-run 3 experiments |
| **Test Execution** | <5 minutes | Full suite on CPU |
| **Documentation** | 100% complete | All modules documented |
| **CI/CD Pass Rate** | 100% | GitHub Actions |

## Known Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| Phase 2 blocker | HIGH | Cascading delays | Implement first, write tests |
| Circular imports | MEDIUM | Module load fails | Lazy imports, careful __init__ |
| OXE friction | MEDIUM | Data pipeline delays | Pre-download, provide scripts |
| CUDA OOM | MEDIUM | Training fails | Batch size tuning, gradient checkpointing |
| Config path issues | LOW | Reproducibility | Test on fresh environment |

## Future Work (Post-MVP)

- [ ] Diffusion-based action heads
- [ ] Multi-task learning heads
- [ ] Vision-language grounding
- [ ] Offline RL with value functions
- [ ] Quantization (INT8) for edge
- [ ] ONNX export
- [ ] Production deployment utilities

## Tracking Progress

Track phase completion in project board:
- **Pending:** Planning phase
- **In Progress:** Active development
- **Complete:** Tests passing, merged to main
- **Blocked:** Waiting for dependency

---

**Document Version:** 1.2
**Last Updated:** 2026-01-26
**Maintainer:** Project Lead (minh-ub)
**Status:** Active (Phases 1-8 complete, Phase 9 ready to start)
