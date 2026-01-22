# Project Roadmap - tinyVLA

## Current Status (2026-01-22)

**Phase:** Initial Scaffolding (Phase 1 Complete)
**Implementation:** 59 LOC (project setup only)
**Timeline:** 2-3 weeks to MVP (with 2 developers)
**Estimated Effort:** 40 hours total

## 12-Phase Bootstrap Plan

### Phase Overview Table

| Phase | Name | Effort | Status | Dependencies | Deliverable |
|-------|------|--------|--------|--------------|-------------|
| **1** | Project Setup | 2h | COMPLETE | None | pyproject.toml, dir structure, logging |
| **2** | Core Registries | 2h | PENDING | Phase 1 | Registry pattern, component factories |
| **3** | NN Primitives | 4h | PENDING | Phase 2 | Attention, MLP, norms, temporal layers |
| **4** | Vision Backbone | 3h | PENDING | Phase 2, 3 | DINOv2, SigLIP, ViT wrappers |
| **5** | Language Backbone | 3h | PENDING | Phase 2, 3 | GPT-2, tokenization |
| **6** | Fusion Mechanisms | 4h | PENDING | Phase 2-5 | Perceiver, cross-attn, concat |
| **7** | Action Heads | 3h | PENDING | Phase 2, 3 | Discrete bins, Gaussian, hybrid |
| **8** | VLA Model | 4h | PENDING | Phase 2-7 | Model orchestration, checkpoint I/O |
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

**Status:** Phase 1 scaffolding complete; ready for Phase 2

---

### Phase 2: Core Registries (BLOCKING)

**Timeline:** Week 1 | **Effort:** 2h
**Priority:** CRITICAL (blocks phases 3-7)
**Dependencies:** Phase 1

**Objectives:**
1. Implement `Registry` base class for component registration
2. Create global registries: VISION_REGISTRY, LANGUAGE_REGISTRY, FUSION_REGISTRY, ACTION_REGISTRY
3. Add factory functions for component instantiation from config

**Deliverables:**
```python
# src/vla/registry/core.py
class Registry:
    """Generic component registry with type safety."""
    def register(name: str) -> decorator
    def get(name: str, **kwargs) -> Component

# src/vla/registry/__init__.py
VISION_REGISTRY = Registry("vision")
LANGUAGE_REGISTRY = Registry("language")
FUSION_REGISTRY = Registry("fusion")
ACTION_REGISTRY = Registry("action")
```

**Success Criteria:**
- [x] Registry class implemented with type hints
- [x] All 4 global registries instantiated
- [x] Unit tests for registration/retrieval
- [x] Zero circular dependencies

**Risk:** Circular imports if __init__ imports all components; use lazy loading

---

### Phases 3-7: Component Implementation (PARALLELIZABLE)

**Timeline:** Weeks 2-3 | **Total Effort:** 17h
**Dependencies:** Phase 2 (all depend on registry)
**Note:** Can parallelize once Phase 2 complete

#### Phase 3: Neural Network Primitives

**Timeline:** Week 2 | **Effort:** 4h

**Components:**
```python
# src/vla/nn/
attention.py       # MultiHeadAttention with Flash Attention 2
mlp.py            # MLP with configurable activation
norms.py          # RMSNorm, LayerNorm
embeddings.py     # RoPE (Rotary Position Embeddings)
temporal.py       # FrameStacker, CausalConv1d
```

**Implementation Details:**
- MultiHeadAttention: Support Flash Attention 2 backend (2-4x speedup)
- MLP: GELU activation, optional dropout
- RMSNorm: Used in modern transformers (pre-norm + RMSNorm)
- RoPE: Efficient position encoding for transformers
- FrameStacker: Stack N temporal frames for action history

**Success Criteria:**
- [x] All primitives implemented with type hints
- [x] Flash Attention support verified
- [x] Unit tests for each component (shapes, gradients)
- [x] <200 LOC per file

---

#### Phase 4: Vision Backbone

**Timeline:** Week 2 | **Effort:** 3h
**Dependencies:** Phase 2, 3

**Components:**
```python
# src/vla/backbones/
vision.py         # VisionBackbone base class
dinov2.py        # DINOv2 (ViT-B/14, primary)
siglip.py        # SigLIP (ViT-B/16, alternative)
```

**Implementation:**
```python
class DINOv2(VisionBackbone):
    """DINOv2 from timm library."""
    def __init__(self, size="base", pretrained=True, freeze=True):
        # Load from timm
        # Freeze if requested
        # Optionally extract intermediate features

    def forward(self, images: Tensor) -> Tensor:
        # Input: [B, 3, 224, 224]
        # Output: [B, N=196, D=768]
```

**Success Criteria:**
- [x] DINOv2 loads from timm correctly
- [x] Freeze flag works (no gradients when frozen)
- [x] Output shape matches expected [B, N, D]
- [x] Tests pass on CPU and GPU

---

#### Phase 5: Language Backbone

**Timeline:** Week 2 | **Effort:** 3h
**Dependencies:** Phase 2, 3

**Components:**
```python
# src/vla/backbones/
language.py       # LanguageBackbone base class
gpt2.py          # GPT-2 encoder
```

**Implementation:**
```python
class GPT2Backbone(LanguageBackbone):
    """GPT-2 language encoder."""
    def __init__(self, model_name="gpt2", freeze=True):
        # Load from transformers
        # Get tokenizer
        # Freeze if requested

    def forward(self, texts: List[str]) -> Tensor:
        # Input: List of strings [B]
        # Tokenize, pad, embed
        # Output: [B, L, D=768]
```

**Success Criteria:**
- [x] GPT-2 loads from transformers correctly
- [x] Tokenization works (handles variable lengths)
- [x] Freeze flag works
- [x] Output shape [B, L, D]

---

#### Phase 6: Fusion Mechanisms

**Timeline:** Week 2-3 | **Effort:** 4h
**Dependencies:** Phase 2-5

**Components:**
```python
# src/vla/fusion/
fusion.py         # FusionModule base class
perceiver.py     # Perceiver Resampler (primary)
cross_attn.py    # Cross-attention (alternative)
concat.py        # Concatenation (baseline)
```

**Implementation (Perceiver):**
```python
class PerceiverResampler(FusionModule):
    """Fixed-size latent bottleneck fusion."""
    def __init__(self, latent_dim=768, num_latents=64, num_layers=4):
        self.latents = nn.Parameter(...)  # Learnable [K, D]
        self.cross_attn_layers = ...      # K layers of cross-attention

    def forward(self, vision: Tensor, language: Tensor) -> Tensor:
        # Input: [B, N, D_v], [B, L, D_l]
        # Cross-attention: latents + (vision || language)
        # Output: [B, K, D]
```

**Success Criteria:**
- [x] Perceiver produces [B, 64, 768] output
- [x] Cross-attention alternatives work
- [x] All versions tested with dummy data
- [x] Gradient flow verified

---

#### Phase 7: Action Heads

**Timeline:** Week 3 | **Effort:** 3h
**Dependencies:** Phase 2, 3

**Components:**
```python
# src/vla/policy/
head.py           # ActionHead base class
discrete.py      # Discrete binning (256 bins per DOF)
continuous.py    # Gaussian (mean + var)
```

**Implementation (Discrete):**
```python
class DiscreteActionHead(ActionHead):
    """256-bin per DOF classification head."""
    def __init__(self, feature_dim=768, action_dim=7, num_bins=256):
        self.mlp = MLP(feature_dim, action_dim * num_bins)

    def forward(self, features: Tensor) -> Tensor:
        # Input: [B, K, D] → global avg pool → [B, D]
        # Output: [B, action_dim, num_bins]

    def loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        # CrossEntropyLoss over bin dimension
```

**Success Criteria:**
- [x] Discrete head outputs [B, 7, 256] logits
- [x] Continuous head outputs [B, 7, 2] (mean, logvar)
- [x] Loss functions work (CrossEntropy, GaussianNLL)
- [x] Gradient flow verified

---

### Phase 8: VLA Model Orchestration

**Timeline:** Week 3 | **Effort:** 4h
**Dependencies:** Phase 2-7

**Deliverables:**
```python
# src/vla/models/
vla.py            # Main VLAModel class
config.py         # VLAConfig dataclass
```

**Implementation:**
```python
@dataclass
class VLAConfig:
    vision_encoder: str = "dinov2"
    language_model: str = "gpt2"
    fusion_type: str = "perceiver"
    action_type: str = "discrete"
    # ... other fields

class VLAModel(nn.Module):
    """Main VLA model orchestrator."""
    def __init__(self, config: VLAConfig):
        self.vision = VISION_REGISTRY.get(config.vision_encoder)
        self.language = LANGUAGE_REGISTRY.get(config.language_model)
        self.fusion = FUSION_REGISTRY.get(config.fusion_type)
        self.action_head = ACTION_REGISTRY.get(config.action_type)

    def forward(self, images: Tensor, texts: List[str]) -> Tensor:
        v = self.vision(images)           # [B, N, D_v]
        l = self.language(texts)          # [B, L, D_l]
        fused = self.fusion(v, l)         # [B, K, D]
        actions = self.action_head(fused) # [B, 7] or [B, 7, 256]
        return actions

    def save_checkpoint(self, path: Path) -> None:
        # Save model + config

    def load_checkpoint(self, path: Path) -> None:
        # Load model + config
```

**Success Criteria:**
- [x] End-to-end forward pass works
- [x] Dummy data produces correct output shape
- [x] Checkpoint save/load works
- [x] Config properly serialized

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

**Document Version:** 1.0
**Last Updated:** 2026-01-22
**Maintainer:** Project Lead (minh-ub)
**Status:** Active (Phase 1 complete, Phase 2 ready to start)
