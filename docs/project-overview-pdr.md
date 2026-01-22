# Project Overview & Product Development Requirements (PDR)

## 1. Problem Statement

Current Vision-Language-Action (VLA) research lacks a unified, modular framework that:
- Supports rapid experimentation with different vision encoders, fusion mechanisms, and action decoders
- Provides production-ready training infrastructure (distributed, tracked, reproducible)
- Enables composition of components via configuration rather than code changes
- Balances research flexibility with implementation simplicity

Existing solutions (OpenVLA, Octo, π0) are often monolithic or use different frameworks (JAX vs PyTorch), making it difficult to benchmark and combine approaches.

## 2. Project Vision

Build a lightweight, composable VLA research framework that:
- Allows researchers to mix-and-match components (vision encoders, fusion mechanisms, action heads) through configuration
- Provides clear abstractions for rapid prototyping without sacrificing production quality
- Supports both behavioral cloning and offline RL training approaches
- Scales from single-GPU research to multi-node distributed training
- Maintains reproducibility via Hydra configuration management

**Success:** A researcher can train a custom VLA model in <2 weeks using tinyVLA.

## 3. Target Users & Use Cases

### Primary Users
- **Robotics Researchers:** Studying manipulation, navigation, dexterous control
- **ML Engineers:** Building production robotics systems
- **Students:** Learning VLA concepts with manageable codebase

### Use Cases

| Use Case | Description | Scope |
|----------|-------------|-------|
| **Benchmarking** | Compare vision encoders (DINOv2 vs SigLIP) or fusion strategies (Perceiver vs cross-attention) | Single GPU, 1-2 week experiments |
| **Rapid Prototyping** | Adapt pretrained VLA to new robot/task without full retraining | Transfer learning, lightweight tuning |
| **Multi-Robot Learning** | Train on Open X-Embodiment dataset combining 22+ robot platforms | Multi-dataset mixing, large-scale training |
| **Production Deployment** | Export trained model for robot inference with latency constraints | Inference optimization, export to ONNX |
| **Policy Ablations** | Study impact of action heads (discrete vs continuous vs diffusion) | Controlled experiments, reproducible configs |
| **Offline RL Extensions** | Add value networks, Q-learning on collected trajectory data | Optional, beyond behavioral cloning baseline |

## 4. Core Requirements

### Functional Requirements

| Requirement | Description | Priority | Target Phase |
|-------------|-------------|----------|--------------|
| **FR-1: Component Registry** | Dynamic component instantiation (vision, fusion, action) via registry | HIGH | Phase 2 |
| **FR-2: Vision Backbones** | Support DINOv2, SigLIP, ViT variants from timm library | HIGH | Phase 4 |
| **FR-3: Language Encoders** | Support GPT-2 (124M-355M) from transformers | HIGH | Phase 5 |
| **FR-4: Fusion Mechanisms** | Perceiver Resampler, cross-attention, concatenation fusion | HIGH | Phase 6 |
| **FR-5: Discrete Action Head** | 256-bin per DOF classification (RT-2 style) | HIGH | Phase 7 |
| **FR-6: Continuous Action Head** | Gaussian distribution (MSE loss) with uncertainty | MEDIUM | Phase 7 |
| **FR-7: Hydra Config System** | Hierarchical configs with CLI overrides and multirun sweeps | HIGH | Phase 9 |
| **FR-8: Data Pipeline** | Load OXE dataset (HDF5 + WebDataset), multi-dataset mixing | HIGH | Phase 10 |
| **FR-9: Training Loop** | PyTorch Lightning with FSDP, mixed precision, WandB logging | HIGH | Phase 11 |
| **FR-10: Checkpointing** | Save/load model weights, config snapshots, training state | HIGH | Phase 11 |
| **FR-11: Multi-GPU Training** | FSDP support for multi-GPU/multi-node scaling | MEDIUM | Phase 11 |
| **FR-12: Test Suite** | Unit tests (registry, components) + integration tests (end-to-end) | HIGH | Phase 12 |

### Non-Functional Requirements

| Requirement | Description | Target | Rationale |
|-------------|-------------|--------|-----------|
| **NFR-1: Performance** | Inference latency: 50-100ms (single GPU inference) | π0 parity (30-50ms) | Real-time robot control |
| **NFR-2: Memory** | Fit full model on single 24GB GPU with batch_size=32 | 16-20GB peak | Accessibility for researchers |
| **NFR-3: Reproducibility** | All experiments reproducible from `.hydra/config.yaml` + seed | 100% of runs | Research rigor |
| **NFR-4: Code Quality** | Black + Ruff lint passing, mypy type checking, 80%+ test coverage | Pre-commit + CI/CD | Maintainability |
| **NFR-5: Documentation** | API docs, examples, architecture diagrams, deployment guide | Completeness > 90% | Developer velocity |
| **NFR-6: Scalability** | Support OXE (1M+ trajectories) without distributed training bottlenecks | <2s epoch overhead | Large-scale research |
| **NFR-7: Extensibility** | Add new vision/fusion/action components without modifying core | Registry pattern | Ecosystem growth |

## 5. Success Criteria & KPIs

### Phase Milestones

| Phase | Success Criteria | Target |
|-------|------------------|--------|
| **1-2** | Project setup + registry pattern complete; all modules importable | Week 1 |
| **3-7** | All components instantiable; forward pass works with dummy data | Weeks 2-3 |
| **8-9** | VLA model orchestrates components; Hydra configs functional | Week 3 |
| **10-11** | Data pipeline loads OXE; Lightning training executes 1 epoch | Week 4 |
| **12** | 80%+ test coverage; all tests pass in <5 min on CPU | Week 4 |

### End-of-Project Success Metrics

| Metric | Target | Method |
|--------|--------|--------|
| **Code Coverage** | 80%+ | pytest --cov report |
| **Inference Latency** | <100ms/sample on single GPU | torch.cuda.max_memory_allocated tracking |
| **Memory Usage** | <24GB peak with batch_size=32 | GPU profiling |
| **Config Reproducibility** | 100% | Re-run 5 experiments with saved configs |
| **Documentation** | All modules documented; README complete | Readability audit |
| **CI/CD Pass Rate** | 100% of commits pass checks | GitHub Actions green |
| **Test Execution Time** | <5 minutes (full suite on CPU) | pytest timing |

## 6. Project Scope

### In Scope (MVP)

1. **Component Registry** — Dynamic instantiation pattern
2. **Neural Network Primitives** — Attention, MLP, norms, temporal layers
3. **Vision/Language Backbones** — DINOv2, GPT-2 wrappers (frozen by default)
4. **Fusion Mechanisms** — Perceiver Resampler, cross-attention
5. **Action Heads** — Discrete binning (primary), Gaussian (alternative)
6. **VLA Model** — Composition of above components
7. **Hydra Configuration** — Hierarchical configs for experiments
8. **Data Pipeline** — Dummy data, HDF5 support, OXE integration
9. **Training Loop** — PyTorch Lightning with WandB tracking
10. **Testing** — Unit + integration tests with 80%+ coverage

### Out of Scope (Post-MVP)

- Diffusion-based action heads (Phase 7 placeholder only)
- Multi-task learning (task-specific heads)
- Quantization / edge deployment
- Vision-language grounding / spatial reasoning
- Offline RL with value functions
- Sim-to-real transfer utilities

## 7. Technical Constraints & Assumptions

### Constraints

| Constraint | Impact | Workaround |
|-----------|--------|-----------|
| **Single GPU Primary** | Development targets 24GB VRAM | Batch size <= 32 |
| **Python 3.10+** | Requires modern Python features | Type hints, match statements |
| **PyTorch 2.5+** | Depends on latest PyTorch API | torch.compile, FSDP2 available |
| **Pretrained Weights Required** | Vision/language encoders frozen | Transfer learning only; full training post-MVP |
| **OXE Data Preprocessing** | RLDS → PyTorch adapter is external | Pre-convert to HDF5 or use WebDataset |
| **WandB Integration** | Requires WandB account for tracking | Optional; can run offline |

### Assumptions

1. **Frozen Backbones** — Assume pretrained vision (DINOv2, SigLIP) and language (GPT-2) encoders are sufficient; don't require finetuning for target tasks
2. **Behavioral Cloning** — Primary training paradigm; offline RL is post-MVP
3. **7-DOF Action Space** — Design targets robotic arms (7D normalized + discretized); extensible to other DOFs
4. **Batch Training** — All training via mini-batches; online/streaming RL is post-MVP
5. **Single Node Primary** — Development optimizes for single GPU; FSDP is scalability feature, not requirement

## 8. Stakeholders & Roles

| Stakeholder | Role | Responsibility |
|-------------|------|-----------------|
| **Project Lead (minh-ub)** | Architecture, coordination | Overall design, phase orchestration |
| **Backend Developer** | Implementation | Implement Phases 3-7 (components) |
| **Infrastructure Engineer** | Training/deployment | Implement Phases 9-11 (config/data/training) |
| **QA/Testing** | Quality assurance | Implement Phase 12 (testing, CI/CD) |

## 9. Timeline & Effort

### Effort Breakdown

| Phase | Effort | Dependencies |
|-------|--------|--------------|
| Phase 1: Setup | 2h | None |
| Phase 2: Registries | 2h | Phase 1 |
| Phase 3: NN Primitives | 4h | Phase 2 |
| Phase 4: Vision Backbone | 3h | Phase 2, 3 |
| Phase 5: Language Backbone | 3h | Phase 2, 3 |
| Phase 6: Fusion | 4h | Phase 2, 3, 4, 5 |
| Phase 7: Action Heads | 3h | Phase 2, 3 |
| Phase 8: VLA Model | 4h | Phase 2-7 |
| Phase 9: Hydra Configs | 4h | Phase 1 |
| Phase 10: Data Pipeline | 5h | Phase 1, 9 |
| Phase 11: Training | 4h | Phase 8, 9, 10 |
| Phase 12: Testing | 2h | Phase 1-11 |
| **Total** | **40 hours** | |

### Timeline (Estimated)

- **Week 1:** Phases 1-2 (setup + registries)
- **Week 2-3:** Phases 3-7 (components, parallel where possible) + Phase 8 (model)
- **Week 3:** Phases 9-10 (configs + data, parallel with component work)
- **Week 4:** Phase 11 (training infrastructure)
- **Week 4:** Phase 12 (testing + final integration)

**Estimated Calendar Time (2 developers):** 2-3 weeks

## 10. Risk Assessment

### High-Risk Areas

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| **Phase 2 Blocker** | HIGH | Phases 3-7 blocked | Implement Phase 2 first, write tests immediately |
| **OXE Integration Friction** | MEDIUM | Data pipeline delays | Pre-download dataset, provide gsutil commands in Phase 10 |
| **CUDA Memory OOM** | MEDIUM | Training fails | Gradient checkpointing, mixed precision, batch size tuning |
| **Circular Imports** | MEDIUM | Module loading fails | Lazy imports, careful `__init__.py` design, tests validate |
| **Hydra Config Path Issues** | LOW | Reproducibility broken | Use absolute paths, test on fresh environment |

### Mitigation Strategies

1. **Early Testing** — Write tests for registry pattern (Phase 2) before implementing components
2. **Documentation** — Document expected APIs before implementation (phases 3-7)
3. **Profiling** — Profile memory early (Phase 11); optimize if approaching limits
4. **CI/CD** — Enable GitHub Actions checks to catch import errors early

## 11. Success Story

### Expected User Journey

```
Researcher Goal: Compare vision encoders on robot manipulation task

1. Install: pip install -e ".[dev]"
2. Data Prep: Download OXE subset (1000 episodes)
3. Config: Create experiment YAML with DINOv2 vs SigLIP
4. Train: python scripts/train.py --multirun model.vision_encoder=dinov2,siglip
5. Analyze: View results in WandB; compare metrics side-by-side
6. Deploy: Export best model; run on robot at 20fps

Total Time: ~1-2 weeks (mostly training time, not coding)
Code Written by Researcher: <100 lines (mostly config YAML)
```

## 12. Future Extensions (Post-MVP)

- Diffusion-based action heads with DDPMScheduler
- Multi-task heads for instruction-following
- Vision-language grounding for spatial reasoning
- Offline RL with value functions / Q-learning
- Sim-to-real transfer utilities
- Model quantization (INT8) for edge deployment
- ONNX export for non-PyTorch environments

## 13. Related Documentation

- **[Code Standards](./code-standards.md)** — Implementation guidelines, code style
- **[System Architecture](./system-architecture.md)** — Component design, data flow
- **[Project Roadmap](./project-roadmap.md)** — Detailed phase breakdown
- **[Tech Stack](./tech-stack.md)** — Technology choices with rationale

---

**Document Version:** 1.0
**Last Updated:** 2026-01-22
**Status:** Active (project in Phase 1 scaffolding)
