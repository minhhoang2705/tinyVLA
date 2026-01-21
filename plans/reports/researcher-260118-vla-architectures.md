# VLA Architecture Research Report
**Date:** 2026-01-18 | **Focus:** SOTA Architectures & Implementation Patterns

---

## Executive Summary
VLA models unified vision, language, and action prediction (2023-2025). Three dominant patterns emerge: (1) **LLM-based** (RT-2, OpenVLA)—leverage pretrained LMs as action decoders; (2) **Diffusion policies** (Octo, π0)—probabilistic action distribution; (3) **Hybrid** (π0 Transfusion)—token fusion across modalities.

---

## 1. Architecture Paradigms

### LLM-as-Action-Decoder (RT-2, OpenVLA)
- **Vision encoder:** DINOv2, SigLIP (ViT variants)
- **Language model:** Llama 2/3, GPT-family
- **Fusion:** Prepend vision tokens to LM input
- **Action output:** Discrete token sequences (e.g., "arm_left gripper_open")
- **Advantage:** Leverages 10B+ token pretraining
- **Limitation:** Quantized action space; slower inference

**OpenVLA specifics (2024):**
- 7B params, beats RT-2-X (55B) by 16.5% on 29-task suite
- Vision: DINOv2 (448²) + SigLIP patches fused via cross-attention
- Action: 256-bin discretization per DOF
- Training: Behavioral cloning on 1M Open X-Embodiment episodes

### Diffusion Policies (Octo)
- **Backbone:** Transformer (27M–93M params)
- **Action prediction:** Diffusion decoder (DDPM variants)
- **Modular attention:** Separate attention heads per modality
- **Temporal:** Observation history (6 frames) → action sequence (8 steps)
- **Advantage:** Continuous smooth actions; excellent generalization
- **Limitation:** Slower inference (~100ms per action)

**Octo specifics:**
- Trained on 800k trajectories; adapts to new morphologies with <1k target demos
- Supports natural language + goal images
- Multi-robot evaluation: ≥90% success on tasks with modest finetuning

### Hybrid Token Fusion (π0/Transfusion)
- **Core:** Treat discrete (language tokens) + continuous (vision, action) as unified token stream
- **Design:** Meta/Waymo Transfusion architecture
- **Advantage:** Seamless cross-modality communication; 7x faster than diffusion baselines
- **Performance (Dec 2024):** "Large improvements" over OpenVLA/Octo on complex tasks (laundry folding, table bussing)

---

## 2. Vision Encoders

| Encoder | Params | Spatial Res | Use Case |
|---------|--------|-------------|----------|
| ViT-B/32 | 86M | 224² | Baseline |
| DINOv2 | 1.1B | 448² | Feature-rich (OpenVLA) |
| SigLIP | 400M | 224² | Multimodal alignment |
| Vision Transformer Large | — | 336² | RT-2 variant |

**Trend (2025):** DINOv2 + SigLIP fusion dominates; avoids language-bias of CLIP.

---

## 3. Action Prediction Mechanisms

### Discrete Bins (RT-2)
```
Action = [arm_x, arm_y, arm_z, gripper, base_yaw, ...]
Each: 256 bins → continuous via percentile decoding
```
- Pros: Simple, stable training
- Cons: Limited precision; post-hoc smoothing needed

### Continuous Gaussian (Diffusion)
```
p(a_t | s_t) = N(μ(s_t), Σ(s_t)) via diffusion steps
```
- Pros: Natural uncertainty; multimodal actions
- Cons: Slower inference (10+ denoising steps)

### Multi-task Action Spaces (Octo, π0)
- Separate prediction heads per task or unified low-rank adapter
- Modular attention: one head per action DOF → fewer parameters
- Enables zero-shot task transfer via task token conditioning

---

## 4. Temporal Modeling (2025 Advances)

**Frame Stacking:** 6-frame history (standard); tested up to 16 frames with diminishing returns

**Visual Trace Prompting (2025):**
- Off-the-shelf point tracker (DINOv2 + RAFT) generates motion traces
- Overlay traces on input as spatial prompts → explicit temporal awareness
- Outperforms naive history concatenation on 8-frame sequences

**TTF-VLA (Temporal Token Fusion):**
- Pixel-attention integration; fuses temporal features without redundancy
- Rolling-window inpainting at inference for continuous trajectory generation

**Recurrent Policies (Emerging):**
- Transformer-XL or Mamba backbones tested; mixed results vs Transformers
- Latency remains challenge for reactive control

---

## 5. Training Strategies

### Behavioral Cloning (BC)
- Baseline for all SOTA models; 1M+ demonstrations
- Open X-Embodiment dataset: 22 robot types, 800k–1M episodes
- Convergence: ~2–4 weeks on A100 cluster

### Offline RL (Emerging 2024)
- Advantage-weighted BC: filter low-Q trajectories before BC
- TD3+BC: policy gradient + BC regularization
- **Challenge:** Offline RL adds complexity with marginal gains (1–2% over BC)

### Online Fine-tuning
- Octo benchmark: <1k target episodes + 4 LoRA adapters → 92% success
- π0: Transfusion enables rapid adaptation with gradient caching

### Key Hyperparameters
- LR: 1e-4 (BC), 1e-5 (RL)
- Batch size: 256–512
- Warmup: 5k steps
- No data augmentation dominates (2024 finding); spatial transforms hurt sim2real

---

## 6. Inference Patterns

| Model | Latency | Throughput | Hardware |
|-------|---------|-----------|----------|
| RT-2 (8B) | 50–100ms | 10 fps | 1x A100 |
| OpenVLA (7B) | 80–120ms | 8 fps | 1x A100 |
| Octo-Base | 100–150ms | 7 fps | 1x A100 |
| π0 (Transfusion) | 30–50ms | 20+ fps | 1x A100/H100 |

**Edge deployment:** Quantized OpenVLA on Jetson Orin (INT8): ~300ms latency

---

## 7. Key Implementation Considerations

1. **Vision aliasing:** Proprietary robot cameras (e.g., Realsense) require domain-specific calibration; Octo/OpenVLA assume 480p color
2. **Action spaces:** Discretization must match robot kinematics; 256-bin sufficient for most manipulators
3. **Multi-embodiment training:** Normalize action spaces to [0,1] before mixing robots
4. **Evaluation:** Success rate insufficient; track trajectory smoothness + energy efficiency
5. **Generalization gap:** SOTA models ≤55% sim-to-real transfer without target domain data; finetuning essential

---

## 8. Unresolved Questions
- Optimal temporal history length for dexterous tasks (beyond 6 frames)?
- How to efficiently leverage language for few-shot learning (π0 suggests it's possible)?
- Is diffusion-based action necessary or is discretized Gaussian sufficient?

---

## Sources
- [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246)
- [RT-2: Vision-Language-Action Models](https://robotics-transformer2.github.io/)
- [Octo: An Open-Source Generalist Robot Policy](https://arxiv.org/abs/2405.12213)
- [π0 Robot Policy Model](https://www.physicalintelligence.company/blog/pi0)
- [Vision-Language-Action Models Survey](https://vla-survey.github.io/)
- [TTF-VLA & Temporal Token Fusion (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/8667f264f88c7938a73a53ab01eb1327-Paper-Conference.pdf)
