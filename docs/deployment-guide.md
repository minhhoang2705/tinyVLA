# Deployment & Inference Guide - tinyVLA

## 1. Environment Setup

### Prerequisites

**Hardware:**
- GPU: NVIDIA RTX 3090 (24GB VRAM) minimum; RTX 4090 or A100 recommended
- CPU: Modern Intel/AMD (8+ cores)
- RAM: 32GB minimum (64GB recommended for OXE dataset)
- Storage: 500GB SSD (for dataset caching)

**Software:**
- Python 3.10+ (3.11+ recommended)
- CUDA 11.8+ (for GPU support)
- NVIDIA cuDNN 8.6+
- pip or conda

### Installation Steps

```bash
# 1. Clone repository
git clone <repo-url>
cd tinyVLA

# 2. Create Python virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Verify Python version
python --version  # Should be 3.10+

# 4. Install package with dependencies
pip install -e ".[dev]"

# 5. Verify installation
python -c "import torch; import vla; print(f'PyTorch: {torch.__version__}, tinyVLA: {vla.__version__}')"

# 6. (Optional) Install CUDA libraries for faster operations
pip install cuda-toolkit

# 7. Verify GPU access (if applicable)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name()}')"
```

### Troubleshooting Installation

**Issue:** CUDA not found
```bash
# Solution: Specify CUDA version during installation
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Issue:** timm models fail to load
```bash
# Solution: Pre-download models
python -c "import timm; timm.create_model('vit_base_patch14_dinov2')"
```

**Issue:** Transformers tokenizer not found
```bash
# Solution: Pre-download transformers models
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('gpt2')"
```

## 2. Training Setup

### Quick Start Training

```bash
# Train with default configuration (dummy data)
python scripts/train.py

# Expected output:
# INFO - Starting training...
# Epoch 1/100: 100%|████████| 100/100 [00:45<00:00, 2.20it/s]
# train/loss: 2.34, val/loss: 2.41
```

### Configuration Overrides

```bash
# Single parameter override
python scripts/train.py train.batch_size=64 train.learning_rate=3e-4

# Nested override
python scripts/train.py model.vision_encoder=siglip model.fusion_type=cross_attn

# Multiple overrides
python scripts/train.py \
  train.batch_size=32 \
  train.learning_rate=1e-4 \
  train.num_epochs=50 \
  data.type=dummy

# Multirun sweep (creates multiple runs)
python scripts/train.py --multirun \
  model.vision_encoder=dinov2,siglip \
  train.learning_rate=1e-4,3e-4

# With WandB project specification
python scripts/train.py \
  hydra/job_logging=colorlog \
  +wandb_config.project=tinyvla_research \
  +wandb_config.entity=your_username
```

### Advanced Training Configuration

**Mixed Precision Training (faster, lower memory):**
```bash
python scripts/train.py train.mixed_precision=true
```

**Gradient Accumulation (simulate larger batches):**
```bash
python scripts/train.py \
  train.batch_size=16 \
  train.accumulation_steps=2  # Effective batch: 32
```

**Multi-GPU Training (FSDP):**
```bash
# Automatic (all GPUs)
python scripts/train.py trainer.devices=-1

# Specific GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train.py trainer.devices=4

# With FSDP strategy
python scripts/train.py trainer.strategy=fsdp trainer.devices=8
```

### Monitoring Training

**View in Terminal:**
```bash
# Tail training logs
tail -f outputs/2026-01-22/10-30-45/train.log
```

**View in WandB Dashboard:**
1. Training automatically logs to WandB (if configured)
2. Visit: https://wandb.ai/your_username/tinyvla
3. Compare runs, inspect metrics, download logs

**GPU Monitoring:**
```bash
# Monitor GPU usage in real-time
watch -n 1 nvidia-smi

# Or use separate terminal
nvidia-smi dmon -s pucm -n 1
```

## 3. Using Open X-Embodiment Dataset

### Dataset Preparation

**Option A: Use Pre-converted HDF5 (Fastest)**

```bash
# Download pre-converted OXE subset (example URL)
wget https://storage.googleapis.com/oxe-data/tinyvla-subset.hdf5
mv tinyvla-subset.hdf5 data/oxe.hdf5

# Train with HDF5
python scripts/train.py data.type=hdf5 data.hdf5_path=data/oxe.hdf5
```

**Option B: Convert from TFRecord (Full OXE)**

```bash
# 1. Install conversion dependencies
pip install tensorflow tensorflow_datasets

# 2. Download OXE dataset (requires gsutil)
gsutil -m cp -r gs://gresearch/robotics-transformer-oxe/data/* data/oxe_raw/

# 3. Convert to HDF5
python scripts/convert_tfrecord_to_hdf5.py \
  --input data/oxe_raw \
  --output data/oxe.hdf5 \
  --subset_size 10000  # Use first 10k episodes

# 4. Train
python scripts/train.py data.type=hdf5 data.hdf5_path=data/oxe.hdf5
```

### Multi-Dataset Mixing

```bash
# Train on mixture of datasets
python scripts/train.py \
  data.type=mixture \
  data.datasets=[dummy,oxe] \
  data.weights=[0.3,0.7]  # 30% dummy, 70% OXE
```

## 4. Model Checkpointing & Resuming

### Save Checkpoints

**Automatic (per epoch):**
```bash
python scripts/train.py \
  trainer.checkpoint_callback.save_top_k=3 \
  trainer.checkpoint_callback.save_last=true
```

**Manual:**
```python
from vla import VLAModel
from pathlib import Path

# After training
model = VLAModel(config)
model.save_checkpoint(Path("checkpoints/my_model.pt"))
```

### Resume from Checkpoint

```bash
# Resume training from latest checkpoint
python scripts/train.py \
  ckpt_path=outputs/2026-01-22/10-30-45/checkpoints/last.ckpt

# Or load specific checkpoint
python scripts/train.py \
  ckpt_path=outputs/2026-01-22/10-30-45/checkpoints/epoch-50.ckpt
```

### Load Pre-trained Weights

```python
from vla import VLAModel
from pathlib import Path

# Load model
model = VLAModel(config)
checkpoint = torch.load("checkpoints/best.pt", map_location="cpu")
model.load_state_dict(checkpoint["model_state"])
model.eval()
```

## 5. Inference

### Single Image Inference

```python
from pathlib import Path
import torch
from PIL import Image
from vla import VLAModel
from vla.utils import setup_logger

logger = setup_logger(__name__)

# Load model
config = {...}  # From config file or dict
model = VLAModel(config)
checkpoint = torch.load("checkpoints/best.pt", map_location="cuda")
model.load_state_dict(checkpoint["model_state"])
model.eval()
model.to("cuda")

# Prepare input
image = Image.open("robot_view.jpg").convert("RGB")
image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
image_tensor = image_tensor.unsqueeze(0).to("cuda")  # [1, 3, 224, 224]

text = "pick up the red cube"

# Inference
with torch.no_grad():
    logits = model(image_tensor, [text])  # [1, action_dim] or [1, action_dim, 256]

# Post-process
if logits.shape[-1] == 256:  # Discrete
    action_bins = logits.argmax(dim=-1)  # [1, 7]
    actions = (action_bins.float() / 255.0) * 2 - 1  # Scale to [-1, 1]
else:  # Continuous
    actions = logits.sigmoid() * 2 - 1  # Scale to [-1, 1]

logger.info(f"Predicted actions: {actions}")
print(f"Action vector: {actions[0].cpu().numpy()}")
```

### Batch Inference

```python
from torch.utils.data import DataLoader
from vla.data import HDF5Dataset

# Load dataset
dataset = HDF5Dataset("data/oxe.hdf5", num_samples=100)
loader = DataLoader(dataset, batch_size=32, num_workers=4)

# Inference loop
all_predictions = []
with torch.no_grad():
    for batch in loader:
        images, texts, _ = batch
        images = images.to("cuda")

        predictions = model(images, texts)
        all_predictions.append(predictions.cpu())

predictions = torch.cat(all_predictions, dim=0)
print(f"Total predictions: {predictions.shape}")
```

### Real-Time Robot Control

```python
import cv2
from collections import deque

class RobotController:
    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device
        self.frame_stack = deque(maxlen=4)

    def get_action(self, frame, instruction):
        """Get action for current frame and instruction."""
        # Preprocess
        frame = cv2.resize(frame, (224, 224))
        frame = torch.from_numpy(frame).float() / 255.0

        # Stack frames (temporal)
        self.frame_stack.append(frame)
        if len(self.frame_stack) < 4:
            return None  # Not enough frames yet

        # Batch frames
        stacked = torch.stack(list(self.frame_stack)).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            logits = self.model(stacked, [instruction])

        # Post-process
        if logits.shape[-1] == 256:
            action = logits.argmax(dim=-1)[0].float() / 255.0 * 2 - 1
        else:
            action = logits.sigmoid()[0] * 2 - 1

        return action.cpu().numpy()

# Usage
controller = RobotController(model)
instruction = "grasp the object"

cap = cv2.VideoCapture(0)  # Webcam
while True:
    ret, frame = cap.read()
    if not ret:
        break

    action = controller.get_action(frame, instruction)
    if action is not None:
        # Send to robot (pseudo-code)
        robot.set_action(action)
        print(f"Action: {action}")

    cv2.imshow("Robot View", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## 6. Performance Optimization

### Inference Speed Optimization

**torch.compile (2-3x speedup):**
```python
model = torch.compile(model)  # JIT compile model
```

**Quantization (4x memory reduction):**
```python
# Convert to INT8
from torch.quantization import quantize_dynamic
model_int8 = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
```

### Memory Optimization

**Gradient Checkpointing:**
```bash
python scripts/train.py train.gradient_checkpointing=true
```

**Mixed Precision:**
```bash
python scripts/train.py train.precision=16  # FP16
```

**Batch Size Tuning:**
```bash
# Find optimal batch size with binary search
python scripts/find_max_batch_size.py  # Utility script
```

## 7. Troubleshooting

### Common Issues

**Issue: CUDA out of memory**
```bash
# Solutions:
1. Reduce batch size: python scripts/train.py train.batch_size=8
2. Enable gradient checkpointing: train.gradient_checkpointing=true
3. Enable mixed precision: train.precision=16
4. Use smaller model: model.vision_encoder=vit_small
```

**Issue: Very slow training**
```bash
# Check GPU utilization:
nvidia-smi dmon -s pucm

# Solutions:
1. Increase batch size (if not OOM)
2. Use num_workers in DataLoader: data.num_workers=4
3. Enable mixed precision: train.precision=16
4. Use faster vision encoder: model.vision_encoder=siglip
```

**Issue: Model produces NaN/Inf**
```bash
# Solutions:
1. Reduce learning rate: train.learning_rate=1e-5
2. Add gradient clipping: train.gradient_clip_val=1.0
3. Check input normalization
4. Verify loss computation
```

**Issue: Poor model performance**
```bash
# Debug checklist:
1. Verify data preprocessing (image normalization, action scaling)
2. Check loss is decreasing: plot training curves in WandB
3. Inspect sample predictions: print first few predictions
4. Try longer training: increase num_epochs
5. Experiment with learning rate schedule
```

## 8. Deployment to Production

### Export for Deployment

**PyTorch Format (native):**
```python
torch.save({
    "model_state": model.state_dict(),
    "config": config,
}, "model.pt")
```

**ONNX Format (framework-agnostic):**
```python
import onnx

dummy_image = torch.randn(1, 3, 224, 224)
dummy_text = ["sample instruction"]

torch.onnx.export(
    model,
    (dummy_image, dummy_text),
    "model.onnx",
    input_names=["images", "texts"],
    output_names=["actions"],
)

# Verify ONNX model
onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)
```

### Docker Container

```dockerfile
# Dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-runtime-ubuntu22.04

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y git
RUN pip install --upgrade pip

# Copy code
COPY . .

# Install tinyVLA
RUN pip install -e .

# Default command
CMD ["python", "scripts/train.py"]
```

**Build and run:**
```bash
docker build -t tinyvla:latest .
docker run --gpus all -v $(pwd)/data:/app/data tinyvla:latest
```

### Cloud Deployment (Example: Google Cloud Run)

```bash
# Build container
gcloud builds submit --tag gcr.io/PROJECT_ID/tinyvla

# Deploy
gcloud run deploy tinyvla \
  --image gcr.io/PROJECT_ID/tinyvla \
  --memory 24Gi \
  --gpu 1 \
  --timeout 3600
```

## 9. Benchmarking

### Inference Latency Benchmark

```python
import time
import torch

def benchmark_inference(model, num_runs=100):
    """Measure inference latency."""
    model.eval()
    model.cuda()

    dummy_image = torch.randn(1, 3, 224, 224).cuda()
    dummy_text = ["test instruction"]

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_image, dummy_text)

    # Benchmark
    torch.cuda.synchronize()
    start = time.time()

    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_image, dummy_text)
            torch.cuda.synchronize()

    elapsed = time.time() - start
    latency_ms = (elapsed / num_runs) * 1000

    print(f"Average latency: {latency_ms:.2f} ms")
    print(f"FPS: {1000 / latency_ms:.1f}")

    return latency_ms

benchmark_inference(model)
```

### Memory Usage Benchmark

```python
import torch

def benchmark_memory(model):
    """Measure peak GPU memory."""
    model.cuda()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    dummy_image = torch.randn(1, 3, 224, 224).cuda()
    dummy_text = ["test"]

    with torch.no_grad():
        _ = model(dummy_image, dummy_text)

    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f"Peak GPU memory: {peak_memory:.0f} MB")

    return peak_memory

benchmark_memory(model)
```

## 10. Best Practices

**Training:**
- Start with dummy data for quick validation
- Use Hydra multirun for hyperparameter search
- Save checkpoints frequently (every N epochs)
- Monitor both train and validation losses
- Use WandB for experiment tracking

**Inference:**
- Batch inputs when possible (vectorization)
- Pre-allocate GPU memory
- Use torch.no_grad() context
- Implement frame stacking for temporal models

**Deployment:**
- Export model to ONNX for portability
- Container-ize with Docker
- Set up monitoring/logging
- Implement graceful degradation (fallback policies)

---

**Last Updated:** 2026-01-22
**Version:** 1.0
