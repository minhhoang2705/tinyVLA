# Tech Stack - tinyVLA

## Core Framework
**PyTorch 2.5+** - Deep learning framework
- **Rationale**: Industry standard for research, excellent flexibility, strong RL ecosystem
- **Features**: torch.compile, FSDP2 for distributed training, Flash Attention integration
- **Compatibility**: CUDA 11.8+, Python 3.10+

## Configuration Management
**Hydra 1.3+** - Hierarchical configuration framework
- **Rationale**: Enables composable configs, experiment tracking, hyperparameter sweeps
- **Benefits**: Runtime overrides, automatic reproducibility, multi-run experiments
- **Integration**: Native support for dataclasses, WandB/MLflow logging

## Vision Backbone
**timm 1.0+** - PyTorch Image Models library
- **Rationale**: 500+ pretrained models, consistent API, PyTorch 2.x optimized
- **Primary models**:
  - DINOv2 (ViT-B/14) - Strong visual representations
  - SigLIP - Vision-language alignment
  - ViT variants - Flexible resolution support
- **Memory**: Flash Attention 2 support for long sequences

## Data Loading
**Open X-Embodiment Dataset**
- **Format**: RLDS (Reverb Dataset) via TFRecord
- **Strategy**: WebDataset tar archives for streaming + local HDF5 cache
- **Preprocessing**:
  - `tf.data` for initial conversion
  - PyTorch IterableDataset for training
  - Multi-dataset mixing with config-driven weights

## Training Infrastructure
**PyTorch Lightning 2.2+** - Training abstraction
- **Rationale**: Reduces boilerplate, handles distributed training, integrates with Hydra
- **Features**: Auto mixed precision, gradient accumulation, FSDP support
- **Callbacks**: Checkpointing, early stopping, learning rate scheduling

**WandB (Weights & Biases)** - Experiment tracking
- **Rationale**: Industry standard, excellent visualization, hyperparameter sweeps
- **Integration**: Native Hydra + Lightning support
- **Features**: Real-time metrics, artifact versioning, collaborative debugging

## Model Components

### Vision Encoder Options
1. **DINOv2 ViT-B/14** (86M params) - Primary choice
   - Pretrained on ImageNet-22k + web data
   - Strong zero-shot transfer
2. **SigLIP ViT-B/16** (87M params) - Alternative
   - Vision-language pretrained
   - Better for instruction following

### Language Model
**GPT-2 Small/Base** (124M-355M params) - Lightweight baseline
- **Rationale**: Sufficient for instruction encoding, well-supported
- **Alternative**: LLaMA 2 (7B) for full VLA (requires more compute)

### Fusion Mechanism
**Perceiver Resampler** - Cross-attention based fusion
- **Rationale**: Fixed-size latent bottleneck, efficient for multi-frame inputs
- **Params**: ~2-8M (minimal overhead)

### Action Head
**Hybrid Approach**:
1. **Discrete Binning** (256 bins per dimension) - Default
   - Classification loss, stable training
   - Follows RT-2/OpenVLA pattern
2. **Continuous Gaussian** - Alternative
   - MSE loss, regression head
   - Better for fine-grained control

## Development Tools
**Core**:
- **Python 3.10+** - Language
- **uv** or **poetry** - Dependency management
- **pytest** - Testing framework
- **black + ruff** - Code formatting + linting
- **mypy** - Type checking

**Debugging**:
- **tensorboard** - Training visualization (fallback to WandB)
- **pdb/ipdb** - Interactive debugging
- **torch.profiler** - Performance profiling

## Compute Requirements
**Minimum**:
- GPU: NVIDIA RTX 3090 (24GB VRAM)
- RAM: 32GB
- Storage: 500GB SSD (for OXE subset)

**Recommended**:
- GPU: NVIDIA A100 (40GB+) or multi-GPU setup
- RAM: 64GB+
- Storage: 2TB NVMe (full OXE dataset)

## Dependencies Summary
```toml
# pyproject.toml core dependencies
python = "^3.10"
torch = "^2.5.0"
torchvision = "^0.20.0"
pytorch-lightning = "^2.2.0"
hydra-core = "^1.3.0"
timm = "^1.0.0"
transformers = "^4.40.0"  # For tokenizers/language models
wandb = "^0.16.0"
einops = "^0.7.0"  # Tensor operations
numpy = "^1.26.0"
pillow = "^10.0.0"
h5py = "^3.10.0"  # HDF5 support
webdataset = "^0.2.86"  # Streaming datasets
tensorboard = "^2.15.0"
```

## Design Decisions

### Why PyTorch over JAX?
- **Ecosystem**: More VLA reference implementations (OpenVLA, Octo-PyTorch port)
- **Debugging**: Easier interactive debugging for research
- **Community**: Larger community, more tutorials
- **Trade-off**: JAX offers better performance at scale but steeper learning curve

### Why Hydra?
- **Composability**: Perfect for VLA's modular architecture (vision/language/fusion/action)
- **Reproducibility**: Auto-saves all configs to `.hydra/` directory
- **Experimentation**: Multi-run sweeps without code changes
- **Trade-off**: Initial learning curve, but pays off for multi-experiment workflows

### Why WebDataset + HDF5?
- **Streaming**: WebDataset enables training on datasets larger than disk
- **Performance**: HDF5 provides fastest local I/O for random access
- **Flexibility**: Can mix both strategies (stream from cloud, cache locally)
- **Trade-off**: More complex setup than pure TFRecord, but better PyTorch integration

### Why Perceiver Resampler over Simple Cross-Attention?
- **Efficiency**: Fixed-size latent bottleneck (e.g., 64 tokens) reduces compute
- **Multi-frame**: Handles variable-length visual sequences elegantly
- **Proven**: Used in Flamingo, RT-2, and other multimodal models
- **Trade-off**: Adds ~2-8M params, but negligible vs vision/language encoders

## Alternative Considerations

### If Compute-Constrained:
- Use ViT-Small (22M) instead of ViT-Base
- Switch to GPT-2 Small (124M) language model
- Reduce batch size, enable gradient accumulation
- Use mixed precision (bfloat16)

### If Targeting Real-time Inference:
- Use torch.compile for 2-3x speedup
- Quantize vision encoder to int8
- Consider distillation from larger model
- Profile and optimize bottlenecks

### If Exploring Diffusion Policies:
- Add **diffusers** library (Hugging Face)
- Replace action head with DDPMScheduler
- Requires 10-100x more inference steps (trade-off)

## Next Steps
1. Create `pyproject.toml` with dependency specifications
2. Set up Hydra config directory structure
3. Implement registry patterns for modular components
4. Create base trainers and data loaders
