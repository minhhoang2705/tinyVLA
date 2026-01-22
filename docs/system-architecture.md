# System Architecture - tinyVLA

## 1. High-Level Architecture Overview

tinyVLA follows a **modular composition pattern** where independent components are assembled into a complete Vision-Language-Action model via Hydra configuration.

```
┌─────────────────────────────────────────────────────────────────┐
│                    VLA Model (Orchestrator)                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                   Forward Pass Pipeline                   │  │
│  │                                                           │  │
│  │  Images [B,3,H,W]              Text [B]                 │  │
│  │     │                             │                      │  │
│  │     ▼                             ▼                      │  │
│  │  ┌─────────────┐          ┌──────────────┐             │  │
│  │  │   Vision    │          │   Language   │             │  │
│  │  │  Backbone   │          │   Backbone   │             │  │
│  │  │ (DINOv2)    │          │  (GPT-2)     │             │  │
│  │  └──────┬──────┘          └──────┬───────┘             │  │
│  │         │                        │                      │  │
│  │    [B,N,D_v]                [B,L,D_l]                  │  │
│  │         │                        │                      │  │
│  │         └────────────┬───────────┘                      │  │
│  │                      │                                  │  │
│  │                      ▼                                  │  │
│  │            ┌────────────────┐                          │  │
│  │            │ Fusion Module  │                          │  │
│  │            │ (Perceiver)    │                          │  │
│  │            └────────┬───────┘                          │  │
│  │                     │                                  │  │
│  │              [B,K,D_fused]                            │  │
│  │                     │                                  │  │
│  │                     ▼                                  │  │
│  │          ┌─────────────────┐                          │  │
│  │          │  Action Head    │                          │  │
│  │          │ (Discrete/      │                          │  │
│  │          │  Continuous)    │                          │  │
│  │          └────────┬────────┘                          │  │
│  │                   │                                   │  │
│  │            [B, action_dim]                           │  │
│  │                   │                                   │  │
│  │                   ▼                                   │  │
│  │            Actions/Logits                            │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 2. Component Interactions

### Data Flow

```
Training/Inference Input
        │
        ├─ Raw Images [B, C=3, H, W]
        │   • Batch of RGB images
        │   • Normalized to [0, 1]
        │   • Shape: typically [B, 3, 224, 224]
        │
        └─ Text Instructions [B]
            • List of strings (e.g., "pick up red cube")
            • Tokenized inside language backbone


Stage 1: Vision Encoding
        │
        ├─ Vision Backbone (frozen DINOv2/timm)
        │   • Input: [B, 3, 224, 224]
        │   • ViT processing (patch embedding + transformers)
        │   • Output: Vision Features [B, N=196, D_v=768]
        │   • Frozen (no gradients) for transfer learning
        │
        └─ Vision Features [B, N, D_v]


Stage 2: Language Encoding
        │
        ├─ Language Backbone (frozen GPT-2/transformers)
        │   • Input: Text → Tokenize → Token IDs [B, L]
        │   • GPT-2 embedding + transformer layers
        │   • Output: Language Features [B, L, D_l=768]
        │   • Frozen (no gradients) for transfer learning
        │
        └─ Language Features [B, L, D_l]


Stage 3: Fusion
        │
        ├─ Fusion Module (Perceiver Resampler - TRAINED)
        │   • Input: Vision [B, N, D_v] + Language [B, L, D_l]
        │   • Cross-attention: Vision/Language → Latent tokens
        │   • Fixed 64 latent tokens (configurable)
        │   • Output: Fused Features [B, K=64, D=768]
        │   • Gradients enabled (only this layer trained)
        │
        └─ Fused Features [B, K, D]


Stage 4: Action Prediction
        │
        ├─ Action Head (trained)
        │   • Discrete Mode: 256 bins per DOF (RT-2 style)
        │     - Linear projection + softmax per DOF
        │     - Output: Logits [B, 7, 256]
        │
        │   • Continuous Mode: Gaussian distribution
        │     - Linear head for mean + log_var
        │     - Output: Mean [B, 7], LogVar [B, 7]
        │
        └─ Actions [B, action_dim]


Training Loss Computation
        │
        ├─ Discrete: CrossEntropyLoss(logits, target_bins)
        │
        └─ Continuous: GaussianNLLLoss(actions, target_actions)
```

## 3. Component Architecture Details

### 3.1 Registry Pattern (IMPLEMENTED)

**Purpose:** Enable dynamic component loading without code modifications. Type-safe, O(1) lookup.

**Implementation Status:** Phase 2 Complete
- Generic `Registry[T]` class in `src/vla/registry/base.py` (157 LOC)
- 5 global instances: VISION_REGISTRY, LANGUAGE_REGISTRY, FUSION_REGISTRY, ACTION_REGISTRY, MODEL_REGISTRY
- Factory functions in `src/vla/registry/factories.py` (138 LOC)
- 20 unit tests with 92% coverage

**Architecture:**
```
┌──────────────────────────────────────────────────────────┐
│              Global Registries (base.py)                  │
├──────────────────────────────────────────────────────────┤
│  VISION_REGISTRY                                         │
│    ├─ register("dinov2_base") → DINOv2Class            │
│    ├─ register("siglip_base") → SigLIPClass            │
│    └─ register("vit_base") → ViTClass                  │
│                                                          │
│  LANGUAGE_REGISTRY                                      │
│    ├─ register("gpt2") → GPT2Class                      │
│    └─ register("gpt2_medium") → GPT2MediumClass        │
│                                                          │
│  FUSION_REGISTRY                                        │
│    ├─ register("perceiver") → PerceiverClass           │
│    ├─ register("cross_attn") → CrossAttention          │
│    └─ register("concat") → ConcatFusion                │
│                                                          │
│  ACTION_REGISTRY                                        │
│    ├─ register("discrete") → DiscreteHeadClass         │
│    └─ register("gaussian") → GaussianHeadClass         │
│                                                          │
│  MODEL_REGISTRY                                         │
│    └─ register("vla_base") → VLAModelClass             │
└──────────────────────────────────────────────────────────┘

                ↓ Factory Functions (factories.py)

        build_vision_encoder(cfg: DictConfig)
        build_language_encoder(cfg: DictConfig)
        build_fusion_module(cfg: DictConfig)
        build_action_head(cfg: DictConfig)
        build_model(cfg: DictConfig)
```

**Registry Class API:**
```python
# Registration (decorator-based)
@VISION_REGISTRY.register("dinov2_base")
class DINOv2(nn.Module):
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        self.hidden_dim = hidden_dim

# Direct instantiation
encoder = VISION_REGISTRY.get("dinov2_base", hidden_dim=1024)

# Get class without instantiating
cls = VISION_REGISTRY.get_class("dinov2_base")

# List available components
components = VISION_REGISTRY.list_available()  # ['dinov2_base', ...]

# Check membership
if "dinov2_base" in VISION_REGISTRY:
    # Component is available
    pass

# Error handling (helpful messages)
try:
    encoder = VISION_REGISTRY.get("unknown")
except KeyError as e:
    # Outputs: "Available components: dinov2_base, ..."
    pass
```

**Factory Function Pattern (for Hydra integration):**
```python
# Registry-based config
cfg = DictConfig({"name": "dinov2_base", "hidden_dim": 768})
encoder = build_vision_encoder(cfg)  # Uses VISION_REGISTRY.get()

# Hydra _target_ based config
cfg = DictConfig({"_target_": "timm.create_model", "model_name": "vit_base_patch16_224"})
encoder = build_vision_encoder(cfg)  # Uses hydra.utils.instantiate()
```

**Usage in Training (post-Phase 8):**
```python
from vla.registry import build_model, build_vision_encoder
from hydra.utils import instantiate

# Load Hydra config
cfg = instantiate_hydra_config()

# Build components via factories
vision = build_vision_encoder(cfg.vision)
language = build_language_encoder(cfg.language)
fusion = build_fusion_module(cfg.fusion)
action = build_action_head(cfg.action)

# Or build complete model at once
model = build_model(cfg.model)
```

### 3.2 Vision Backbone

**Supported Models:**
- **DINOv2 (Primary):** Self-supervised ViT, 1.1B params total (224² spatial)
  - Size: Base (86M params), Large (300M params)
  - Frozen during training for transfer learning
  - Extracted to intermediate layer if needed

- **SigLIP (Alternative):** Vision-language aligned, 400M params
  - Size: Small, Base, Large
  - Multimodal pretraining on web data
  - Better instruction following than pure vision models

**Interface:**
```python
class VisionBackbone(nn.Module):
    """Base interface for vision encoders."""

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, 3, H, W] normalized to [0, 1]

        Returns:
            features: [B, num_patches, feature_dim]
                     e.g., [B, 196, 768] for ViT-B/14
        """

# Instantiation (frozen):
encoder = DINOv2(size="base", pretrained=True, freeze=True)
features = encoder(images)  # No gradients computed
```

### 3.3 Language Backbone

**Supported Models:**
- **GPT-2:** Baseline language model (124M-355M params)
  - Small (124M): Faster, less memory
  - Base (355M): Recommended balance
  - Pretrained on English text, good for instruction following

**Interface:**
```python
class LanguageBackbone(nn.Module):
    """Language encoder with tokenization."""

    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        Args:
            texts: List of instruction strings [B]

        Returns:
            features: [B, max_seq_len, feature_dim]
                     e.g., [B, 64, 768] for GPT-2-base
        """
        # Internally:
        # 1. Tokenize texts with GPT-2 tokenizer
        # 2. Pad to max length
        # 3. Pass through GPT-2 transformer layers
        # 4. Return last hidden state
```

### 3.4 Fusion Mechanism

**Primary: Perceiver Resampler**

```
Vision Features [B, N_v=196, D_v=768]
Language Features [B, N_l=64, D_l=768]
          │
          ├─ Project to common dimension (if needed)
          │
          ├─ Create learnable latent queries [K=64, D=768]
          │
          ├─ Cross-attention layer:
          │   Query: Latent queries [K, D]
          │   Key/Value: Vision + Language concatenated [N_v+N_l, D]
          │   Output: Updated latents [K, D]
          │
          ├─ Transformer blocks (typically 4-8 layers)
          │   • Multi-head self-attention on latents
          │   • Cross-attention back to vision+language
          │   • MLP blocks
          │
          └─ Fused Features [B, K=64, D=768]

Advantage:
- Fixed-size bottleneck (K latent tokens) regardless of input length
- Efficient: Compute grows as O(K) not O(N²)
- Proven in Flamingo, RT-2, OpenVLA

Alternative: CrossAttentionFusion
- Direct cross-attention between vision and language
- Lower computational cost but higher memory
- Used in early VLA models

Alternative: ConcatFusion
- Simple concatenation of vision+language features
- Baseline for ablation studies
```

### 3.5 Action Heads

**Discrete Action Head (Primary)**

```
Fused Features [B, K=64, D=768]
          │
          ├─ Global average pooling
          │  → [B, D=768]
          │
          ├─ Linear layer to action logits
          │  → [B, 7*256] (for 7-DOF, 256 bins per DOF)
          │
          ├─ Reshape to [B, 7, 256]
          │
          ├─ Per-DOF softmax (over 256 bins)
          │  → Probability distribution per DOF
          │
          └─ During inference:
             - Argmax or sample from distribution
             - Convert bin index to action range [-1, 1]
             - Action = (bin_index / 255) * 2 - 1

Loss: CrossEntropyLoss
    • Input: Logits [B, 7, 256], Targets [B, 7] (bin indices)
    • Efficient classification loss
    • Stable training
```

**Continuous Action Head (Alternative)**

```
Fused Features [B, K=64, D=768]
          │
          ├─ Global average pooling
          │  → [B, D=768]
          │
          ├─ Linear layers
          │  ├─ Mean head → [B, 7]
          │  └─ LogVar head → [B, 7]
          │
          └─ Gaussian distribution: N(mean, exp(logvar))

Loss: GaussianNLLLoss
    • Likelihood: -log(N(y|mean, var))
    • Captures both prediction and uncertainty
    • Slower convergence than discrete
```

## 4. Configuration System (Hydra)

### Config Hierarchy

```
configs/
├── config.yaml           # Main entry point
│   # Defaults: which subdirectory configs to use
│   defaults:
│     - model: vla         # Load configs/model/vla.yaml
│     - vision: dinov2     # Load configs/vision/dinov2.yaml
│     - language: gpt2     # Load configs/language/gpt2.yaml
│     - fusion: perceiver
│     - action: discrete
│     - train: default
│     - data: dummy
│
├── model/
│   ├── vla.yaml          # Main VLA model config
│   └── vla_large.yaml    # Larger variant
│
├── vision/
│   ├── dinov2.yaml       # DINOv2-base (primary)
│   ├── siglip.yaml       # SigLIP-base
│   └── vit.yaml          # ViT-base
│
├── language/
│   ├── gpt2.yaml         # GPT-2 small
│   └── gpt2_medium.yaml  # GPT-2 base
│
├── fusion/
│   ├── perceiver.yaml    # Perceiver Resampler
│   ├── cross_attn.yaml   # Cross-attention
│   └── concat.yaml       # Concatenation
│
├── action/
│   ├── discrete.yaml     # 256-bin classification
│   └── gaussian.yaml     # Continuous Gaussian
│
├── train/
│   ├── default.yaml      # Default training params
│   └── distributed.yaml  # Multi-GPU settings
│
├── data/
│   ├── dummy.yaml        # Random dummy data
│   ├── oxe.yaml          # Open X-Embodiment
│   └── hdf5.yaml         # Local HDF5
│
└── experiment/
    ├── baseline.yaml     # Default experiment
    └── ablation_*.yaml   # Specific ablations
```

### Config Structure

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

# Model parameters (from model/vla.yaml)
model:
  name: "vla"
  hidden_dim: 768

# Vision encoder (from vision/dinov2.yaml)
vision_encoder:
  type: "dinov2"
  size: "base"
  freeze: true

# Language model (from language/gpt2.yaml)
language_model:
  type: "gpt2"
  model_name: "gpt2"
  freeze: true

# Fusion (from fusion/perceiver.yaml)
fusion:
  type: "perceiver"
  num_latents: 64
  latent_dim: 768
  num_layers: 4

# Action head (from action/discrete.yaml)
action_head:
  type: "discrete"
  action_dim: 7
  num_bins: 256

# Training (from train/default.yaml)
train:
  batch_size: 32
  learning_rate: 1e-4
  num_epochs: 100
  optimizer: "adam"
  warmup_steps: 1000

# Data (from data/dummy.yaml)
data:
  type: "dummy"
  num_samples: 1000
  image_size: 224
```

### Runtime Composition

```bash
# Use default config
python scripts/train.py

# Override single param
python scripts/train.py train.batch_size=64

# Override nested config group
python scripts/train.py vision=siglip language=gpt2_medium

# Multiple overrides
python scripts/train.py model=vla_large train.batch_size=64 train.learning_rate=3e-4

# Multirun sweep (create multiple runs)
python scripts/train.py --multirun \
  vision=dinov2,siglip \
  fusion=perceiver,cross_attn \
  train.lr=1e-4,3e-4
```

## 5. Data Pipeline Architecture

### Data Flow

```
Raw Data Sources
        │
        ├─ Open X-Embodiment (RLDS format)
        │   └─ WebDataset → streaming tar archives
        │
        ├─ Local HDF5 (pre-converted)
        │   └─ Direct h5py access
        │
        └─ Dummy Dataset (for testing)
            └─ Random tensor generation


Data Loading Layer
        │
        ├─ DatasetMixture: Combine multiple datasets
        │   • Config-driven weights
        │   • Per-dataset preprocessing
        │   • Action normalization
        │
        └─ IterableDataset (PyTorch)
            • Streaming batches during training
            • No fixed size
            • Async prefetching


Data Preprocessing
        │
        ├─ Image:
        │   • Resize to 224x224
        │   • Normalize [0,1] → [-1,1]
        │   • Optional augmentation (disabled by default)
        │
        ├─ Text:
        │   • Tokenization (handled by language backbone)
        │
        └─ Actions:
        │   • Continuous [-1, 1] → Discrete [0, 255] (if discrete head)
        │   • Per-dataset normalization (mean/std)


DataLoader (PyTorch)
        │
        └─ Batches [B, 3, 224, 224], [B], [B, 7]
```

## 6. Training Infrastructure (PyTorch Lightning)

### Training Loop Structure

```
VLALightningModule (nn.Module)
        │
        ├─ forward(images, texts)
        │   └─ VLA model forward pass → actions
        │
        ├─ training_step(batch)
        │   ├─ Compute loss (CrossEntropy or GaussianNLL)
        │   ├─ Log metrics (WandB)
        │   └─ Return loss
        │
        ├─ validation_step(batch)
        │   ├─ Compute validation loss
        │   ├─ Optional: Action MAE vs ground truth
        │   └─ Log metrics
        │
        └─ Trainer (Lightning)
            ├─ Handle distributed training (FSDP, DDP)
            ├─ Mixed precision (AMP)
            ├─ Checkpointing
            ├─ Early stopping
            └─ WandB integration
```

### Multi-GPU Training (FSDP)

```
Multi-GPU Setup
        │
        ├─ Single Node: DDP (Data Distributed Parallel)
        │   • Each GPU gets full model copy + different data batch
        │   • Gradients averaged across GPUs
        │
        └─ Multi-Node: FSDP (Fully Sharded Data Parallel)
            • Model parameters sharded across GPUs
            • Each GPU computes full forward pass
            • All-reduce gradient during backward
            • Memory efficient: N GPUs → N/K memory per GPU

PyTorch Lightning handles FSDP automatically:
    strategy = FSDPStrategy()
    trainer = Trainer(strategy=strategy, devices=4)
```

## 7. Module Dependencies

### Dependency Graph

```
registry/ (no dependencies)
    ↓
┌─────────────────────────────────┐
│  nn/  backbones/  fusion/       │ (each depend on registry)
│  policy/                        │
└─────────────────────────────────┘
    ↓
models/ (depends on all above + registry)
    ↓
training/ (depends on models + utils)
    ↓
data/ (depends on utils)
    ↓
train.py (depends on all modules)
```

### Circular Dependency Prevention

```python
# ✓ Lazy imports in __init__.py
if TYPE_CHECKING:
    from vla.models import VLAModel  # Type hint only

# ✓ Import at function level
def get_model() -> "VLAModel":
    from vla.models import VLAModel
    return VLAModel(...)
```

## 8. Storage & Checkpointing

### Model Checkpointing

```
checkpoints/
├── epoch-00.pt          # Periodic checkpoints
├── epoch-01.pt
├── best.pt              # Best validation checkpoint
└── final.pt             # Final model

Checkpoint Contents:
{
    "model_state": {...},       # VLAModel weights
    "optimizer_state": {...},   # Optimizer state
    "epoch": 5,
    "global_step": 1000,
    "config": {...},            # Full Hydra config
    "metrics": {                # Training metrics
        "loss": 0.45,
        "val_loss": 0.52,
    }
}
```

### Config Reproducibility

```
outputs/
├── 2026-01-22/
    └── 10-30-45/
        ├── .hydra/
        │   └── config.yaml      # Exact config used for run
        ├── train.log
        └── checkpoints/
            ├── best.pt
            └── final.pt

Reproducibility:
    python scripts/train.py \
        --config-path outputs/2026-01-22/10-30-45/.hydra \
        --config-name config
```

## 9. Performance Optimization Strategy

### Memory Optimization

1. **Frozen Backbones:** No gradients for vision/language
2. **Gradient Checkpointing:** Trade compute for memory in transformer layers
3. **Mixed Precision:** FP16 for forward pass, FP32 for loss/backward
4. **Batch Size Tuning:** Start with 32, reduce if OOM

### Inference Optimization

1. **torch.compile:** 2-3x speedup on fixed shapes
2. **Quantization:** INT8 for production (post-MVP)
3. **Knowledge Distillation:** Smaller student model (post-MVP)

### Distributed Training

1. **FSDP:** Shard model parameters across GPUs
2. **Gradient Accumulation:** Simulate larger batches
3. **Async Data Loading:** Prefetch next batch during compute

---

**Document Version:** 1.0
**Last Updated:** 2026-01-22
**Status:** Active (architectural blueprint complete)
