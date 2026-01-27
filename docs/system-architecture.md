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

## 2. VLA Model Orchestration (Phase 8 - IMPLEMENTED)

### VLAModel Architecture

The `VLAModel` class orchestrates all components via registry-based composition:

```
VLAConfig (dataclass)
    ├─ VisionConfig (model_name, pretrained, freeze, hidden_dim)
    ├─ LanguageConfig (model_name, pretrained, freeze, hidden_dim)
    ├─ FusionConfig (type, num_latents, latent_dim, num_layers)
    └─ ActionConfig (type, action_dim, num_bins, pooling_type)
           ↓
    VLAModel.__init__()
           ├─ build_vision() → VISION_REGISTRY.get()
           ├─ build_language() → LANGUAGE_REGISTRY.get()
           ├─ build_fusion() → FUSION_REGISTRY.get()
           └─ build_action_head() → ACTION_REGISTRY.get()
           ↓
    Assembled VLA Model
```

**Key Features:**
- **Config-driven composition:** All components selected via VLAConfig
- **Registry integration:** Factory pattern for dynamic instantiation
- **Frozen backbones:** Vision/Language encoders frozen (99% of params)
- **Trainable layers:** Only fusion module and action head trained
- **Dual-mode inference:** Training mode (with loss) and predict mode (inference only)
- **Checkpoint persistence:** Save/load complete state with config
- **Input validation:** Automatic shape and type checking
- **Temporal support:** Optional multi-frame processing via FrameStacker

**Parameter Breakdown (typical configuration):**
```
Total: ~500M parameters
├─ Vision backbone (frozen): 86M (DINOv2-base)
├─ Language backbone (frozen): 124M (GPT-2-small)
├─ Fusion module (trainable): 285M (Perceiver, 4 layers)
└─ Action head (trainable): 5M (discrete, 7-DOF)

Trainable: 290M (58%)
Frozen: 210M (42%)
```

### Training vs Inference

**Training Mode:**
```python
model.train()
output = model(images, texts=instructions, target_actions=actions)
loss = output["loss"]
loss.backward()  # Backprop only through fusion + action head
optimizer.step()
```

**Inference Mode:**
```python
model.eval()
with torch.no_grad():
    actions = model.predict(images, texts=instructions)
    # actions shape: [B, 7] (discrete: argmax over bins)
    # actions shape: [B, 7] (continuous: mean of Gaussian)
```

---

## 3. Component Interactions

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

## 4. Component Architecture Details

### 3.1 Neural Network Primitives (IMPLEMENTED - Phase 3)

**Purpose:** Provide modular, reusable building blocks for transformer-based VLA architectures. All components use PyTorch and support distributed training.

**Implementation:** 832 LOC across 6 files with 70 unit tests achieving 99.5% coverage.

**Key Characteristics:**
- Flash Attention 2 support for 2-4x speedup on compatible hardware
- Type-safe with full docstrings (NumPy style)
- ~100 tests per component module (99.5% coverage)
- No external dependencies beyond PyTorch + einops

**Attention Mechanisms:**

```
┌─────────────────────────────────────────────────┐
│        Attention Mechanisms (attention.py)       │
├─────────────────────────────────────────────────┤
│                                                 │
│  MultiHeadAttention                             │
│  ├─ Self-attention (within-modality)            │
│  ├─ Input: [B, seq_len, dim]                   │
│  ├─ Parallel heads (num_heads=12 typical)      │
│  ├─ Flash Attention when available             │
│  └─ Output: [B, seq_len, dim] (same shape)     │
│                                                 │
│  CrossAttention                                 │
│  ├─ Cross-modal attention (query ≠ key/value) │
│  ├─ Query: [B, Q_len, query_dim]              │
│  ├─ Context: [B, KV_len, context_dim]        │
│  ├─ Use: Language queries → Vision features    │
│  └─ Output: [B, Q_len, query_dim]             │
└─────────────────────────────────────────────────┘
```

**Feed-Forward Networks:**

```
┌─────────────────────────────────────────────────┐
│   Feed-Forward Networks (mlp.py)                │
├─────────────────────────────────────────────────┤
│                                                 │
│  MLP (Standard)                                 │
│  └─ Linear(d → hidden) → Activation           │
│     → Dropout → Linear(hidden → d)             │
│                                                 │
│  GatedMLP (Modern variant)                      │
│  └─ Projects 3x to hidden                      │
│     → Splits: value + gate                     │
│     → Output = value * sigmoid(gate)           │
│     → Lower complexity than attention          │
└─────────────────────────────────────────────────┘
```

**Normalization Strategies:**

```
┌─────────────────────────────────────────────────┐
│   Normalization (norm.py)                       │
├─────────────────────────────────────────────────┤
│                                                 │
│  RMSNorm (Recommended)                          │
│  ├─ RMS-based (faster than LayerNorm)          │
│  ├─ No learnable bias                          │
│  ├─ More stable for large models               │
│  └─ Used in: T5, LLaMA, modern VLAs            │
│                                                 │
│  LayerNorm (Alternative)                        │
│  ├─ Mean-variance based (original Transformer)│
│  └─ Learnable scale & bias                     │
│                                                 │
│  get_norm() Factory                             │
│  └─ Config-driven norm selection               │
└─────────────────────────────────────────────────┘
```

**Positional Encodings:**

```
┌─────────────────────────────────────────────────┐
│   Position Encodings (pos_encoding.py)          │
├─────────────────────────────────────────────────┤
│                                                 │
│  Sinusoidal (Classical)                         │
│  ├─ Fixed patterns: sin(pos/10000^(i/d))      │
│  ├─ No parameters                              │
│  ├─ Excellent extrapolation                    │
│  └─ Token positional info via frequency        │
│                                                 │
│  Learnable (Adaptive)                           │
│  ├─ Trainable embeddings [max_seq, dim]       │
│  ├─ Optimized for specific distribution       │
│  ├─ Limited extrapolation                      │
│  └─ Used in: Original Transformer variants     │
│                                                 │
│  Rotary (Modern - RoPE)                         │
│  ├─ Rotates Q,K in 2D subspaces               │
│  ├─ θ = m * θ₀ where m = position            │
│  ├─ Excellent length extrapolation             │
│  ├─ O(d) memory per token                      │
│  └─ Used in: LLaMA, GPT-3.5+, modern VLAs     │
└─────────────────────────────────────────────────┘
```

**Temporal Processing:**

```
┌─────────────────────────────────────────────────┐
│   Temporal Modeling (temporal.py)               │
├─────────────────────────────────────────────────┤
│                                                 │
│  FrameStacker                                   │
│  ├─ Input: [B, num_frames, 3, H, W]           │
│  ├─ Modes: concat, mean, attention aggregation│
│  ├─ Output: [B, 3*frames, H, W] or [B, 3, H, W]
│  └─ Use: Multi-frame optical flow, motion      │
│                                                 │
│  CausalConv1d                                   │
│  ├─ 1D convolution with causal padding         │
│  ├─ No future leakage (t can't see t+1)       │
│  ├─ Pattern: Conv(dilation) → Norm → Activation
│  └─ Use: Sequential action modeling            │
│                                                 │
│  TemporalBlock (Residual unit)                  │
│  ├─ CausalConv1d → Norm → ReLU → Dropout      │
│  ├─ Residual: out = input + block(input)      │
│  └─ Stack multiple for deeper temporal model   │
└─────────────────────────────────────────────────┘
```

**Integration with VLA Pipeline:**

```
Vision Images [B, 3, H, W]
        ↓
[Optional FrameStacker for multi-frame]
        ↓
Vision Backbone → Features [B, N, D_v]
        │
        ├─ Internal attention: MultiHeadAttention
        └─ Internal MLP layers: MLP or GatedMLP

Language Text [B]
        ↓
Language Backbone → Features [B, L, D_l]
        │
        ├─ Internal attention: MultiHeadAttention
        └─ Internal MLP layers: MLP or GatedMLP

        ↓
Fusion Module
        ├─ CrossAttention (language → vision)
        ├─ MultiHeadAttention (self-fusion)
        ├─ MLP (non-linear mixing)
        └─ RMSNorm (layer normalization)

        ↓
Fused Features [B, K, D]
        │
        ├─ Action Head (linear projection)
        └─ Optional TemporalBlock (sequence modeling)

        ↓
Actions [B, action_dim]
```

**Testing Coverage:**
- 70 unit tests across 5 modules
- Test fixture: device, batch_size, seq_length, tensors
- Coverage: 99.5% (almost all code paths)
- Scenarios: Single GPU, different batch sizes, edge cases

**When to Use Each Component:**
| Component | Use When | Avoid When |
|-----------|----------|-----------|
| MultiHeadAttention | Processing single modality (vision or language) | Need cross-modal fusion (use CrossAttention) |
| CrossAttention | Fusing vision + language | Within-modality processing (use MultiHeadAttention) |
| MLP | Standard feed-forward layer | Need gating mechanism (use GatedMLP) |
| GatedMLP | Want lower complexity than attention | Standard activation sufficient (use MLP) |
| RMSNorm | Building modern transformers | Using legacy LayerNorm code (migrate to RMSNorm) |
| SinusoidalPE | Unknown sequence lengths | Fixed, known lengths (use LearnablePositionEncoding) |
| RoPE | Long-context modeling or extrapolation | Short fixed-length sequences |
| FrameStacker | Multi-frame visual input (optical flow) | Single-frame images |
| CausalConv1d | Temporal action sequences (no future leakage) | Non-temporal 1D data |

**Example: Building a Simple Transformer Block**

```python
from vla.nn import MultiHeadAttention, MLP, RMSNorm

class TransformerBlock(nn.Module):
    def __init__(self, dim: int = 768, num_heads: int = 12, hidden_factor: int = 4):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim=dim, num_heads=num_heads)
        self.norm2 = RMSNorm(dim)
        self.mlp = MLP(dim=dim, hidden_dim=dim*hidden_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm with residual connections (modern pattern)
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
```

---

### 3.2 Registry Pattern (IMPLEMENTED)

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

### 3.2 Vision Backbone (IMPLEMENTED - Phase 4)

**Supported Models:**
- **DINOv2 (Primary):** Self-supervised ViT, Meta
  - Size: Base (86M params), Large (300M params)
  - Frozen during training for transfer learning
  - Extracted to intermediate layer if needed
  - Output: [B, 196, 768] for ViT-B/14

- **SigLIP (Alternative):** Vision-language aligned, OpenAI
  - Size: Small, Base, Large
  - Multimodal pretraining on web data
  - Better instruction following than pure vision models
  - Output: [B, num_patches, feature_dim]

- **ViT (Generic):** HuggingFace timm wrapper
  - Flexible model names via timm library
  - Supports any timm vision model

**Interface:**
```python
from vla.backbones import DINOv2Backbone

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
encoder = DINOv2Backbone(size="base", pretrained=True, freeze=True)
features = encoder(images)  # [B, 196, 768], no gradients
```

**Implementation Details:**
- All models loaded via timm or transformers
- Freeze flag disables gradient computation
- Type-safe with full docstrings
- Registered via VISION_REGISTRY for dynamic loading

### 3.3 Language Backbone (IMPLEMENTED - Phase 5)

**Supported Models:**
- **GPT-2:** Baseline language model
  - Small (124M): Faster, less memory
  - Base (355M): Recommended balance
  - Pretrained on English text, good for instruction following

- **LLaMA:** Modern language model (ALTERNATIVE)
  - 7B-70B params (various sizes)
  - Better long-context modeling
  - Improved instruction following

**Interface:**
```python
from vla.backbones import GPT2Backbone

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

# Instantiation (frozen):
encoder = GPT2Backbone(model_name="gpt2", freeze=True)
features = encoder(texts)  # [B, seq_len, 768], no gradients
```

**Implementation Details:**
- Integrated tokenization (handles padding, truncation)
- Automatic sequence length handling
- Freeze flag disables gradient computation
- Registered via LANGUAGE_REGISTRY
- Handles variable-length instruction sequences

### 3.4 Fusion Mechanism (IMPLEMENTED - Phase 6)

**Primary: Perceiver Resampler**

```
Vision Features [B, N_v=196, D_v=768]
Language Features [B, N_l=64, D_l=768]
          │
          ├─ Create learnable latent tokens [K=64, D=768]
          │   • Initialize as learnable parameters
          │   • Repeated for each batch item
          │
          ├─ Perceiver Stack (4-8 layers, default 4):
          │   Each layer:
          │   ├─ Cross-attention: Latents (Q) attend to Vision+Language (K,V)
          │   ├─ Self-attention: Latents (Q,K,V) self-attention
          │   ├─ MLP: Feed-forward layer with residual
          │   └─ Layer norm: RMSNorm for stability
          │
          ├─ Multiple iterations of cross+self attention
          │   • Each layer refines latent understanding
          │   • Total compute: O(K * L) where L = num_layers
          │
          └─ Fused Features [B, K=64, D=768]

Advantages:
- Fixed-size bottleneck (K=64 tokens) regardless of input length
- Efficient: Compute O(K) not O(N²)
- Proven in Flamingo (80B params), RT-2, OpenVLA
- Flexible: K adjustable for speed/capacity tradeoff

Alternative 1: CrossAttentionFusion
- Direct cross-attention between vision and language
- Output: [B, N_v, D] (keeps all vision patches)
- Lower compression but more information preserved
- Used in earlier VLA models

Alternative 2: ConcatFusion
- Simple concatenation of vision+language features
- Output: [B, N_v + N_l, D]
- Baseline for ablation studies
- Requires downstream head to handle variable length

Alternative 3: AdapterFusion
- Low-rank adapter networks (~1% params)
- Output: [B, K, D]
- Parameter-efficient for resource constraints
```

### 3.5 Action Heads (IMPLEMENTED - Phase 7)

**Discrete Action Head (Primary - RT-2 style)**

```
Fused Features [B, K=64, D=768]
          │
          ├─ Global average pooling (or max pooling)
          │  → [B, D=768]
          │
          ├─ Optional: Projection layer (if D ≠ feature_dim)
          │  → [B, feature_dim]
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
    • Stable training (no gradient explosion)
    • Used in: RT-2, Open Vocabulary robotics models

Advantages:
- Stable gradient flow (classification vs regression)
- Explicit bin representation enables semantic understanding
- Easy to add constraints (e.g., max/min velocities per bin)
```

**Continuous Action Head (Alternative)**

```
Fused Features [B, K=64, D=768]
          │
          ├─ Global average pooling
          │  → [B, D=768]
          │
          ├─ Two linear heads
          │  ├─ Mean head → [B, 7]
          │  └─ LogVar head → [B, 7] (log variance)
          │
          └─ Gaussian distribution: N(mean, exp(logvar))

Loss: GaussianNLLLoss
    • Likelihood: -log(N(y|mean, var))
    • Captures both prediction and uncertainty
    • Slower convergence than discrete
    • Better for continuous dynamics

Advantages:
- Direct continuous predictions
- Uncertainty quantification (log variance)
- Better for smooth action trajectories
```

**Hybrid Action Head (Mixed)**

```
Fused Features [B, K=64, D=768]
          │
          ├─ Split into arm + gripper
          │
          ├─ Arm (6 DOF): Discrete head
          │   └─ Output: [B, 6, 256]
          │
          ├─ Gripper (1 DOF): Continuous head
          │   └─ Output: [B, 1, 2] (mean, logvar)
          │
          └─ Combined loss: 0.6*L_discrete + 0.4*L_continuous

Usage:
- Real robots often need discrete arm + continuous gripper
- Gripper force/position benefits from uncertainty
- Arm joints typically benefit from discrete control
```

## 5. Configuration System (Hydra)

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

## 6. Data Pipeline Architecture

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

## 7. Training Infrastructure (PyTorch Lightning)

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

## 8. Module Dependencies

### Dependency Graph

```
registry/ (no dependencies)
    ↓ (all components depend on registry)
    ├─ nn/ (IMPLEMENTED - no circular deps)
    │  └─ depends: torch, einops, utils.logging
    │
    ├─ backbones/ (IMPLEMENTED - Phase 4-5)
    │  └─ depends: nn, registry, timm/transformers
    │
    ├─ fusion/ (IMPLEMENTED - Phase 6)
    │  └─ depends: nn, registry, torch
    │
    └─ policy/ (IMPLEMENTED - Phase 7)
       └─ depends: nn, registry, torch

    ↓
models/ (IMPLEMENTED Phase 8 - depends on backbones, fusion, policy, registry)
    ├─ vla_base.py (orchestrator, 509 LOC)
    ├─ vla_configs.py (config dataclasses, 186 LOC)
    └─ policy.action_utils (loss computation)

    ↓
training/ (PENDING Phase 11 - depends on models, utils, pytorch-lightning)
    ↓
data/ (PENDING Phase 10 - depends on utils)
    ↓
train.py (depends on all modules)
```

**Circular Dependency Prevention:**
- `nn/` has zero dependencies on other vla modules (only torch, einops)
- `registry/` has zero dependencies (base infrastructure)
- Other modules depend on `nn/` and `registry/` but not vice versa
- Type hints use `TYPE_CHECKING` for forward references
- Lazy imports at function level (not module level)

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

## 9. Storage & Checkpointing

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

## 10. Performance Optimization Strategy

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

**Document Version:** 1.3
**Last Updated:** 2026-01-26
**Status:** Active (Phases 2-8 complete, VLA orchestration implemented)
