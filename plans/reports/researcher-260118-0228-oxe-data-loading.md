# Open X-Embodiment Dataset & Data Loading Research Report

## Dataset Overview

**Scale & Diversity**
- 1M+ trajectories across 60+ pre-existing datasets
- 22 robot platforms (single-arm, bi-manual, quadrupeds)
- 527 distinct manipulation skills
- 34 research institutions, 21 organizations
- Avg trajectory: ~120 timesteps @ 3-10 Hz control frequency

**Task Coverage**: Manipulation, navigation, dexterous control across embodiments

---

## Data Format: RLDS & Episode Structure

**RLDS (Reinforcement Learning Datasets)**
- Serialized TFRecord files (protobuf)
- Hierarchical: Dataset → Episodes → Steps
- Each episode = `tf.data.Dataset` of steps
- Step fields: `is_first`, `is_last`, `observation`, `action`, `reward`, `discount`

**Canonical Observation/Action Schema**
- **Observations**: RGB workspace + wrist camera views (resized to common resolution)
- **Language**: Task instruction field (string)
- **Actions**: Normalized 7D vector (x,y,z, roll,pitch,yaw, gripper) → discretized to 256 bins
- **State**: Robot-specific state vector

**Storage**: TFRecord (primary), HDF5, Parquet/Arrow support

---

## PyTorch Integration Strategies

### Strategy 1: TensorFlow Dataset → PyTorch IterableDataset
```python
# Use tf.data natively with PyTorch
import tensorflow_datasets as tfds
dataset = tfds.load('dataset_name')
# Wrap with custom IterableDataset adapter
# Manual episode iteration over TF dataset
```
- **Pros**: Direct access to RLDS format, no conversion overhead
- **Cons**: TF dependency, requires custom wrapping logic

### Strategy 2: WebDataset Format
- Store data as tar archives (no conversion needed)
- Implements PyTorch IterableDataset interface
- Compatible with standard DataLoader
- Supports streaming without unpacking
- **Performance**: Similar to native format, scales well to large datasets

### Strategy 3: HDF5 Conversion
- Convert RLDS → HDF5 (faster I/O for read-heavy workloads)
- **Benchmarks**: HDF5 ≈ Zarr performance; both outperform direct TFRecord reads
- **Trade-off**: One-time conversion cost vs. training speedup
- Use h5py with PyTorch Dataset for per-episode access

### Strategy 4: Hybrid Streaming + Caching
- Stream episodes from cloud (GCS), cache locally
- Use `gsutil` for bulk downloads: `gsutil -m cp -r gs://gdm-robotics-open-x-embodiment/{dataset}/`
- Combine with prefetching for I/O overlap

---

## Data Loading Best Practices

**Multi-Dataset Mixing**
- Register datasets via config (observation/action spaces, transforms)
- Define mixtures with sampling weights (e.g., OpenVLA "Open-X Magic Soup++": 970K trajectories)
- Use `mixtures.py` pattern for reproducible dataset combinations

**Preprocessing Pipeline**
- Standardization: Normalize actions per dataset before mixing
- Image augmentation: Optional online augmentation during training
- Episode filtering: Remove invalid/failed trajectories
- Step batching: Pad/truncate to fixed sequence length

**Distributed Training**
- PyTorch FSDP (Fully Sharded Data Parallel) compatible
- Shard episodes across workers via IterableDataset
- Avoid duplicates: Each worker processes disjoint episode subsets

**Performance Tips**
- Use `num_workers > 0` in DataLoader for async I/O
- Enable prefetching (tf.data or webdataset)
- Batch operations at step-level (inside episodes) before torch.stack
- Memory-efficient: Use SkipDecoding for lazy image loading
- Cloud datasets: Set retry logic for transient failures

---

## Storage & Performance Considerations

| Format | Read | Write | Cloud | Streaming |
|--------|------|-------|-------|-----------|
| TFRecord | Slow | Fast | ✓ (GCS) | ✓ |
| HDF5 | Fast | Fast | ✗ | ✗ |
| Zarr | Fast | Medium | ✓ | ~ |
| WebDataset | Fast | Medium | ✓ | ✓ |

**Recommendations**
- **Streaming from cloud**: TFRecord + GCS (native OXE setup)
- **On-premise, fixed data**: HDF5 for max throughput
- **Hybrid**, cloud + local: WebDataset + tar archives
- **Multi-GPU training**: WebDataset + IterableDataset (better scaling)

---

## Example Loading Patterns

### OpenVLA Pattern
- RLDS format mandatory
- Dataset config: `configs.py` (spaces), `transforms.py` (preprocessing), `mixtures.py` (composition)
- PyTorch FSDP distributed training
- Configurable batch size, gradient accumulation, image augmentation

### Minimal Custom Loader
1. Use `tfds.load()` to fetch dataset
2. Wrap episode iterator in IterableDataset
3. Implement `__iter__()` yielding steps or episode batches
4. DataLoader with `num_workers=0` (TF not fork-safe) or use ProcessPoolExecutor

---

## Unresolved Considerations

1. **Optimal batch formation**: Step-level vs. episode-level batching trade-offs for sequence models?
2. **Cross-dataset action normalization**: Best practices for aligning embodiment-agnostic actions?
3. **Streaming inference**: Efficient online episode sampling during multi-epoch training?
4. **Memory footprint**: Image caching strategies for 1M+ trajectory dataset?

---

## References

- [Open X-Embodiment GitHub](https://github.com/google-deepmind/open_x_embodiment)
- [OXE Paper (arXiv)](https://arxiv.org/abs/2310.08864)
- [RLDS GitHub](https://github.com/google-research/rlds)
- [OpenVLA Implementation](https://github.com/openvla/openvla)
- [WebDataset Library](https://github.com/webdataset/webdataset)
- [PyTorch DataLoader Docs](https://pytorch.org/docs/stable/data.html)
- [HDF5 vs Zarr Performance](https://arxiv.org/pdf/2207.09503)
