# Performance & VRAM Optimization Action Plan (Target: RTX 4070Ti)

## 🔴 Critical (Quality Impact)
1. **Missing ImageNet Normalization**: Add `transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])` and `antialias=True` to `_process_image()` in `src/vla/data/lerobot_dataset.py`.
2. **Missing Training Augmentation**: Add `RandomResizedCrop(224, scale=(0.75, 1.0))` and `ColorJitter(0.3, 0.3, 0.2, 0.05)` to the train dataset pipeline in `src/vla/data/datamodule_lightning.py`. Do not apply to val/test.

## 🟡 High (Speed / Throughput Bottleneck)
3. **CPU Tokenization Bottleneck**: Tokenization is currently happening inside the GPU forward pass in `language.py`. Move it to `Dataset.__getitem__` or `collate_fn` to prevent CPU-GPU sync stalls.
4. **Missing Torch Compile**: Apply `torch.compile(..., backend="inductor")` to `model.fusion` and `model.action_head` in `lightning_module.py`.
5. **Frame-by-frame Forward Pass**: `TemporalVLAModel.forward()` uses a Python for-loop (`[self.vision(img) for img in image_sequence]`). Replace this with a batched `torch.cat` forward pass to maximize GPU utilization.

## 🟢 Medium (VRAM Optimization)
6. **Missing torch.no_grad()**: Wrap the frozen vision and language backbone forward passes in `with torch.no_grad():` inside `vla_base.py`.
7. **Gradient Checkpointing**: Enable `torch.utils.checkpoint.checkpoint` for `PerceiverBlock` layers in `fusion/perceiver.py` to save VRAM.