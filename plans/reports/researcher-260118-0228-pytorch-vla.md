# PyTorch VLA Implementation Research Report

**Date:** 2026-01-18
**Focus:** Production-ready patterns for Vision Language Action models

---

## 1. PyTorch 2.x Ecosystem

### FSDP (Fully Sharded Data Parallel)
- **FSDP2 vs FSDP1**: FSDP2 uses DTensor-based dim-0 per-parameter sharding (simpler, lower memory)
- **Application Strategy**: Apply `fully_shard()` bottom-up (layers before root model)
- **Performance**: Implicit prefetching with CPU thread for all-gather operations
- **Memory Optimization**: Mixed precision + CPU offloading + selective sharding granularity
- **Checkpoint**: Use PyTorch DCP for both single/multi-node training

### torch.compile
- **Requirements**: Consistent input shapes for best speedup (recompiles on shape changes)
- **Modes**: `default` (balanced), `reduce-overhead` (faster), `max-autotune` (slowest compile)
- **Trade-off**: Early iterations slower due to compilation; subsequent iterations faster
- **Recommendation**: Use with fixed-size batches; avoid dynamic shapes

### Activation Checkpointing
- **Memory Trade-off**: Reduces memory to O(√n) from O(n) at cost of recompute
- **Implementation**: Save only inputs during forward; rematerialize activations in backward
- **Selective Approach**: Checkpoint expensive layers (attention, FFN) selectively

### Flash Attention 2
- **Memory**: O(N) vs O(N²) standard attention (50-80% reduction)
- **Speed**: 2-4x acceleration for long sequences
- **Production Status**: Stable; integrated in transformers library with `use_flash_attn2=True`

---

## 2. Model Building Patterns

### Registry Pattern
```python
class ModelRegistry:
    _registry = {}

    @classmethod
    def register(cls, name):
        def wrapper(model_cls):
            cls._registry[name] = model_cls
            return model_cls
        return wrapper

    @classmethod
    def get(cls, name, **kwargs):
        return cls._registry[name](**kwargs)

@ModelRegistry.register("vla_base")
class VLABase(nn.Module):
    pass
```

**Benefits**: Dynamic loading, config-driven instantiation, clean separation of concerns

### Factory Pattern
- Centralized model creation with consistent weight handling
- Configuration management (model size, architecture variants)
- Hub integration (HuggingFace)
- **Example**: `timm.create_model("vit_base_patch16_224", pretrained=True)`

### Composition over Inheritance
- Modular vision/language/action heads
- Flexible component swapping (ViT, CNN, Mamba backbones)
- Better testability and maintainability

---

## 3. Vision Backbone Integration (timm)

### Current State (2025)
- **Largest Collection**: 500+ architectures (ViT, ResNet, EfficientNet, Swin, ConvNeXt, Mamba)
- **PyTorch 2.x Support**: torch.compile via `--torchcompile` flag
- **Latest ViT**: NaFlexViT with variable aspect ratio/resolution support
- **Hybrid Models**: MambaVision (Mamba + Transformer) on top of timm

### Integration Pattern
```python
import timm

# Load pretrained backbone
backbone = timm.create_model(
    'vit_base_patch16_224',
    pretrained=True,
    num_classes=0,  # Remove classification head
    features_only=True
)

# Freeze backbone, train only head
for param in backbone.parameters():
    param.requires_grad = False
```

### Recommendation for VLA
- **Lightweight**: Vision Transformer (ViT) for action models (27M-93M params)
- **Efficient**: DeiT or MobileViT for on-robot deployment
- **Flexible**: ConvNeXt as hybrid alternative with better FLOPs

---

## 4. Efficient Training Pipeline

### Mixed Precision (AMP)
```python
from torch.amp import autocast, GradScaler

scaler = GradScaler()
for batch in dataloader:
    with autocast(dtype=torch.float16):
        loss = model(batch)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Impact**: 2x faster, 50% memory reduction

### Gradient Accumulation
- Simulate larger batch sizes without OOM
- Effective batch = micro_batch × num_accumulation_steps
- Delay optimizer step until accumulation complete

### Distributed Training Hierarchy
1. **Single GPU**: Standard training
2. **Multi-GPU (single node)**: DDP + AMP
3. **Multi-node**: FSDP2 + AMP + Selective checkpointing
4. **At-scale**: FSDP2 + FSDP + Gradient accumulation + CPU offloading

---

## 5. Popular VLA Implementations

### OpenVLA (Production-Ready)
- **Models**: 1B-34B parameters on PyTorch FSDP
- **Architecture**: Text encoder + lightweight CNN + diffusion decoder
- **Training**: Flash-Attention + FSDP for distributed training
- **Integration**: HuggingFace transformers compatible
- **Status**: Production-ready, actively maintained

### Octo (UC Berkeley)
- **Size**: 27M & 93M parameters (lightweight)
- **Approach**: Text + image encoders + diffusion policy
- **Output**: Continuous joint trajectories (smoother than autoregressive)
- **Performance**: Matches RT-2 (55B) with 100x fewer params
- **Dataset**: Trained on 800k demonstrations from OpenX-Embodiment

### Best Practices from Reference Implementations
1. **Modular architecture**: Separate vision/language/action heads
2. **Efficient backbones**: ViT for vision, small language models for text
3. **Diffusion policies**: Better than autoregressive for continuous control
4. **Dataset scaling**: 800k+ demonstrations improve generalization
5. **Inference optimization**: Use torch.export or ONNX for deployment

---

## 6. Production Checklist

- [ ] Use FSDP2 for multi-GPU; DDP for single-node multi-GPU
- [ ] Enable Flash Attention in transformer layers
- [ ] Apply gradient checkpointing to transformer blocks
- [ ] Use mixed precision (AMP) with GradScaler
- [ ] Freeze vision backbone; fine-tune only action head
- [ ] Consistent input shapes for torch.compile reuse
- [ ] Validate on in-distribution + out-of-distribution splits
- [ ] Export to ONNX/TorchScript for production inference
- [ ] Profile memory/compute with `torch.profiler`

---

## 7. Memory Profile Estimates

| Component | Baseline | Flash Attn | Checkpointing | Combined |
|-----------|----------|-----------|----------------|----------|
| Attention | 100% | 20% | - | 20% |
| Activation | 100% | - | 50% | 50% |
| **Total** | **100%** | **-20%** | **-50%** | **-70%** |

---

## 8. Compatibility Notes

- **PyTorch**: 2.0+ recommended (2.x for compile/FSDP2)
- **Vision Models**: timm 0.9.0+ for modern architectures
- **Distributed**: Requires torch.distributed for DDP/FSDP
- **Hardware**: GPU with CUDA compute capability 7.0+ for Flash Attention
- **Dependencies**: transformers 4.36+, xformers optional (Flash Attention fallback)

---

## Unresolved Questions

1. Should we implement custom diffusion decoder or use existing libraries?
2. Which language model to use for text encoding (LLaMA, BERT, custom)?
3. How to optimize for robot compute (e.g., Jetson boards)?
4. What dataset/domain to focus on for initial release?

---

## Sources

- [PyTorch FSDP Tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [FSDP Advanced Tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_advanced_tutorial.html)
- [PyTorch Activation Checkpointing Blog](https://pytorch.org/blog/activation-checkpointing-techniques/)
- [torch.compile Tutorial](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
- [OpenVLA GitHub](https://github.com/openvla/openvla)
- [OpenVLA Paper](https://arxiv.org/abs/2406.09246)
- [PyTorch Image Models (timm)](https://github.com/huggingface/pytorch-image-models)
- [VLA Concepts Review Paper](https://arxiv.org/html/2505.04769v1)
- [Model Registry Pattern Guide](https://www.abhik.xyz/articles/registry-pattern)
- [timm Documentation](https://timm.fast.ai/)
