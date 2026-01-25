# VLA Model Architecture Diagrams

This document visualizes different Vision-Language-Action (VLA) model architecture combinations using Mermaid.js diagrams. Each diagram shows the complete data flow through components.

---

## 1. Complete VLA Pipeline Overview

This diagram shows the high-level data flow through the VLA system with all major components.

```mermaid
---
theme: default
---
flowchart TB
    %% Input Layer
    subgraph inputs["Input Layer"]
        IMG["Images<br/>[B, 3, 224, 224]"]
        TXT["Text Instructions<br/>[B] strings"]
    end

    %% Vision Processing
    subgraph vision["Vision Encoder (FROZEN)"]
        VB["Vision Backbone<br/>DINOv2 / SigLIP / ViT"]
        VOUT["Vision Features<br/>[B, N=196, D=768]"]
    end

    %% Language Processing
    subgraph language["Language Encoder (FROZEN)"]
        LB["Language Model<br/>GPT-2 / BERT"]
        LOUT["Language Features<br/>[B, L, D=768]"]
    end

    %% Fusion Layer
    subgraph fusion["Fusion Module (TRAINED)"]
        FM["Perceiver Resampler<br/>or<br/>Cross-Attention Fusion"]
        FOUT["Fused Features<br/>[B, K=64, D=768]"]
    end

    %% Action Prediction
    subgraph action["Action Head (TRAINED)"]
        AH["Discrete / Gaussian /<br/>Hybrid Action Head"]
        AOUT["Actions<br/>[B, action_dim=7]"]
    end

    %% Data Flow
    IMG --> VB
    VB --> VOUT
    TXT --> LB
    LB --> LOUT
    VOUT --> FM
    LOUT --> FM
    FM --> FOUT
    FOUT --> AH
    AH --> AOUT

    %% Styling
    classDef frozen fill:#e1f5ff,stroke:#0288d1,stroke-width:3px
    classDef trained fill:#fff3e0,stroke:#f57c00,stroke-width:3px
    classDef data fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px

    class vision,language frozen
    class fusion,action trained
    class IMG,TXT,VOUT,LOUT,FOUT,AOUT data
```

---

## 2. Architecture 1: DINOv2 + GPT-2 + Perceiver + Discrete Actions

The most common configuration used in RT-2 and OpenVLA.

```mermaid
---
theme: default
---
flowchart TD
    %% Input
    IMG["📷 Images<br/>[B, 3, 224, 224]<br/>RGB normalized"]
    TXT["💬 Instructions<br/>['pick up red block',<br/>'move to target']"]

    %% Vision Backbone
    DINO["🔒 DINOv2 Base<br/>vit_base_patch14_dinov2<br/>Frozen Backbone<br/>embed_dim=768"]
    VPATCH["Vision Patches<br/>[B, 196, 768]<br/>14x14 = 196 patches"]

    %% Language Backbone
    GPT2["🔒 GPT-2<br/>Frozen Language Model<br/>embed_dim=768<br/>max_length=77"]
    LTOKEN["Language Tokens<br/>[B, L≤77, 768]<br/>Mean pooled → [B, 1, 768]"]

    %% Fusion
    PERC["🎓 Perceiver Resampler<br/>num_latents=64<br/>num_layers=2<br/>num_heads=8"]
    CONCAT["Concatenate<br/>[B, 196+1=197, 768]"]
    LATENT["Learnable Queries<br/>[B, 64, 768]"]
    XATTN["Cross-Attention Layers<br/>Queries attend to<br/>Vision+Language Context"]
    FUSED["Fixed-Size Latents<br/>[B, 64, 768]"]

    %% Action Head
    POOL["Mean Pool<br/>[B, 768]"]
    DISC["🎓 Discrete Action Head<br/>action_dim=7<br/>num_bins=256"]
    LOGITS["Logits<br/>[B, 7, 256]"]
    BINS["Bin Predictions<br/>argmax → [B, 7]"]
    ACTIONS["Actions<br/>[B, 7]<br/>Range: [-1, 1]"]

    %% Data Flow
    IMG --> DINO
    DINO --> VPATCH
    TXT --> GPT2
    GPT2 --> LTOKEN

    VPATCH --> CONCAT
    LTOKEN --> CONCAT
    CONCAT --> PERC

    LATENT --> XATTN
    CONCAT --> XATTN
    PERC -.contains.- LATENT
    PERC -.contains.- XATTN
    XATTN --> FUSED

    FUSED --> POOL
    POOL --> DISC
    DISC --> LOGITS
    LOGITS --> BINS
    BINS --> ACTIONS

    %% Styling
    classDef frozen fill:#bbdefb,stroke:#1976d2,stroke-width:3px,color:#000
    classDef trained fill:#ffe0b2,stroke:#f57c00,stroke-width:3px,color:#000
    classDef data fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000
    classDef process fill:#fff9c4,stroke:#f9a825,stroke-width:2px,color:#000

    class DINO,GPT2 frozen
    class PERC,DISC trained
    class IMG,TXT,VPATCH,LTOKEN,FUSED,ACTIONS data
    class CONCAT,LATENT,XATTN,POOL,LOGITS,BINS process
```

---

## 3. Architecture 2: SigLIP + GPT-2 + Cross-Attention Fusion + Gaussian Actions

Language-conditioned vision with continuous action predictions.

```mermaid
---
theme: default
---
flowchart TD
    %% Input
    IMG["📷 Images<br/>[B, 3, 224, 224]"]
    TXT["💬 Instructions<br/>[B] text strings"]

    %% Vision
    SIG["🔒 SigLIP Base<br/>vit_base_patch16_siglip<br/>Frozen<br/>embed_dim=768"]
    VPATCH["Vision Patches<br/>[B, 196, 768]"]

    %% Language
    GPT2["🔒 GPT-2<br/>Frozen<br/>embed_dim=768"]
    LSEQ["Language Sequence<br/>[B, L, 768]<br/>output_mode='all'"]

    %% Fusion - Cross Attention
    CROSS["🎓 Cross-Attention Fusion<br/>num_layers=4<br/>num_heads=8"]

    subgraph crosslayers["Cross-Attention Layers (×4)"]
        SELF["Self-Attention<br/>Vision tokens attend<br/>to each other"]
        XATT["Cross-Attention<br/>Vision (Q) attends to<br/>Language (K, V)"]
        FFN["Feed-Forward Network<br/>MLP with GELU"]
    end

    COND["Conditioned Vision<br/>[B, 196, 768]<br/>Language-aware features"]

    %% Action Head
    POOL["Mean Pool Patches<br/>[B, 768]"]
    GAUSS["🎓 Gaussian Action Head<br/>action_dim=7<br/>Predicts μ, σ"]
    MEAN["Mean<br/>[B, 7]"]
    STD["Std Dev<br/>[B, 7]"]
    SAMPLE["Sample if Training<br/>μ + σ * ε"]
    ACTIONS["Actions<br/>[B, 7]<br/>Range: [-1, 1]"]

    %% Flow
    IMG --> SIG
    SIG --> VPATCH
    TXT --> GPT2
    GPT2 --> LSEQ

    VPATCH --> CROSS
    LSEQ --> CROSS
    CROSS -.contains.- SELF
    CROSS -.contains.- XATT
    CROSS -.contains.- FFN
    SELF --> XATT
    XATT --> FFN
    FFN --> COND

    COND --> POOL
    POOL --> GAUSS
    GAUSS --> MEAN
    GAUSS --> STD
    MEAN --> SAMPLE
    STD --> SAMPLE
    SAMPLE --> ACTIONS

    %% Styling
    classDef frozen fill:#bbdefb,stroke:#1976d2,stroke-width:3px
    classDef trained fill:#ffe0b2,stroke:#f57c00,stroke-width:3px
    classDef data fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef process fill:#fff9c4,stroke:#f9a825,stroke-width:2px

    class SIG,GPT2 frozen
    class CROSS,GAUSS trained
    class IMG,TXT,VPATCH,LSEQ,COND,ACTIONS data
    class SELF,XATT,FFN,POOL,MEAN,STD,SAMPLE process
```

---

## 4. Architecture 3: Generic ViT + Language Encoder + Gated Fusion + Hybrid Actions

Flexible configuration with mixed discrete/continuous actions.

```mermaid
---
theme: default
---
flowchart TD
    %% Input
    IMG["📷 Images<br/>[B, 3, 224, 224]"]
    TXT["💬 Instructions"]

    %% Vision
    VIT["🔒 Generic ViT<br/>timm.create_model<br/>vit_base_patch16_224<br/>Frozen"]
    VCLS["CLS + Patches<br/>[B, 197, 768]<br/>output_mode='both'"]

    %% Language
    LANG["🔒 Language Encoder<br/>backend='bert'<br/>Frozen<br/>embed_dim=768"]
    LMEAN["Mean Pooled<br/>[B, 1, 768]<br/>output_mode='mean'"]

    %% Gated Fusion
    GATE["🎓 Gated Fusion<br/>Learnable modality weighting"]

    subgraph gatelayer["Gating Mechanism"]
        EXPAND["Expand Language<br/>[B, 197, 768]<br/>to match vision length"]
        CONCAT["Concatenate<br/>[Vision || Language]<br/>[B, 197, 1536]"]
        SIGMOID["Linear + Sigmoid<br/>gate = σ(W·concat)"]
        BLEND["Weighted Blend<br/>out = gate⊙vision +<br/>(1-gate)⊙language"]
    end

    FUSED["Fused Features<br/>[B, 197, 768]"]

    %% Hybrid Action Head
    POOL["Mean Pool<br/>[B, 768]"]

    subgraph hybrid["🎓 Hybrid Action Head"]
        DHEAD["Discrete Head<br/>6 DOF arm joints<br/>num_bins=256"]
        CHEAD["Continuous Head<br/>1 DOF gripper<br/>Gaussian μ,σ"]
    end

    DACTIONS["Discrete Actions<br/>[B, 6]<br/>Binned predictions"]
    CACTIONS["Continuous Actions<br/>[B, 1]<br/>Gaussian samples"]
    COMBINE["Concatenate<br/>[B, 7]"]
    ACTIONS["Final Actions<br/>[B, 7]"]

    %% Flow
    IMG --> VIT
    VIT --> VCLS
    TXT --> LANG
    LANG --> LMEAN

    VCLS --> GATE
    LMEAN --> GATE
    GATE -.contains.- EXPAND
    LMEAN --> EXPAND
    EXPAND --> CONCAT
    VCLS --> CONCAT
    CONCAT --> SIGMOID
    SIGMOID --> BLEND
    VCLS --> BLEND
    EXPAND --> BLEND
    BLEND --> FUSED

    FUSED --> POOL
    POOL --> hybrid
    hybrid -.contains.- DHEAD
    hybrid -.contains.- CHEAD
    DHEAD --> DACTIONS
    CHEAD --> CACTIONS
    DACTIONS --> COMBINE
    CACTIONS --> COMBINE
    COMBINE --> ACTIONS

    %% Styling
    classDef frozen fill:#bbdefb,stroke:#1976d2,stroke-width:3px
    classDef trained fill:#ffe0b2,stroke:#f57c00,stroke-width:3px
    classDef data fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef process fill:#fff9c4,stroke:#f9a825,stroke-width:2px

    class VIT,LANG frozen
    class GATE,hybrid trained
    class IMG,TXT,VCLS,LMEAN,FUSED,ACTIONS,DACTIONS,CACTIONS data
    class EXPAND,CONCAT,SIGMOID,BLEND,POOL,DHEAD,CHEAD,COMBINE process
```

---

## 5. Temporal Architecture: Multi-Frame VLA with Temporal Perceiver

Processes video sequences for temporally-aware action prediction.

```mermaid
---
theme: default
---
flowchart TD
    %% Input
    FRAMES["📹 Video Frames<br/>[T=8 frames]<br/>Each: [B, 3, 224, 224]"]
    TXT["💬 Instruction<br/>'continuously stir the pot'"]

    %% Vision per frame
    subgraph visionframes["Vision Encoding (Per Frame)"]
        F0["Frame t=0"]
        F1["Frame t=1"]
        F7["Frame t=7"]
    end

    VIT["🔒 Shared Vision Encoder<br/>DINOv2 Base<br/>Frozen"]

    VPATCH0["Patches [B,196,768]"]
    VPATCH1["Patches [B,196,768]"]
    VPATCH7["Patches [B,196,768]"]

    %% Temporal Embedding
    TEMP["🎓 Temporal Embeddings<br/>Learned per time step<br/>embed[t] ∈ ℝ^768"]

    ADD0["Patches + temp[0]"]
    ADD1["Patches + temp[1]"]
    ADD7["Patches + temp[7]"]

    CONCAT_T["Concatenate Temporal<br/>[B, T×196=1568, 768]"]

    %% Language
    GPT["🔒 GPT-2<br/>Frozen"]
    LANG["Language [B, L, 768]"]

    %% Temporal Perceiver
    TPERC["🎓 Temporal Perceiver Resampler<br/>num_latents=64<br/>Handles temporal context"]

    CONTEXT["Vision (all frames) +<br/>Language Context<br/>[B, 1568+L, 768]"]
    LATENTS["Cross-Attention<br/>Latents attend to<br/>full temporal context"]
    FUSED["Temporal Latents<br/>[B, 64, 768]"]

    %% Action
    POOL["Mean Pool<br/>[B, 768]"]
    HEAD["🎓 Discrete Action Head<br/>action_dim=7"]
    ACTIONS["Actions<br/>[B, 7]"]

    %% Flow
    FRAMES --> visionframes
    F0 --> VIT
    F1 --> VIT
    F7 --> VIT

    VIT --> VPATCH0
    VIT --> VPATCH1
    VIT --> VPATCH7

    TEMP --> ADD0
    TEMP --> ADD1
    TEMP --> ADD7

    VPATCH0 --> ADD0
    VPATCH1 --> ADD1
    VPATCH7 --> ADD7

    ADD0 --> CONCAT_T
    ADD1 --> CONCAT_T
    ADD7 --> CONCAT_T

    TXT --> GPT
    GPT --> LANG

    CONCAT_T --> CONTEXT
    LANG --> CONTEXT
    CONTEXT --> TPERC
    TPERC -.contains.- LATENTS
    LATENTS --> FUSED

    FUSED --> POOL
    POOL --> HEAD
    HEAD --> ACTIONS

    %% Styling
    classDef frozen fill:#bbdefb,stroke:#1976d2,stroke-width:3px
    classDef trained fill:#ffe0b2,stroke:#f57c00,stroke-width:3px
    classDef data fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef process fill:#fff9c4,stroke:#f9a825,stroke-width:2px

    class VIT,GPT frozen
    class TEMP,TPERC,HEAD trained
    class FRAMES,TXT,VPATCH0,VPATCH1,VPATCH7,LANG,FUSED,ACTIONS data
    class F0,F1,F7,ADD0,ADD1,ADD7,CONCAT_T,CONTEXT,LATENTS,POOL process
```

---

## 6. Component Registry Pattern

Shows how components are assembled via the registry system.

```mermaid
---
theme: default
---
flowchart LR
    %% Registries
    subgraph registries["Component Registries"]
        VREG["VISION_REGISTRY<br/>dinov2, siglip, timm_vit"]
        LREG["LANGUAGE_REGISTRY<br/>gpt2, bert, distilbert"]
        FREG["FUSION_REGISTRY<br/>perceiver_resampler<br/>cross_attention_fusion<br/>gated_fusion"]
        AREG["ACTION_REGISTRY<br/>discrete_action<br/>gaussian_action<br/>hybrid_action"]
    end

    %% Config
    CONFIG["Configuration<br/>(Hydra YAML)"]

    subgraph config_example["Example Config"]
        VC["vision:<br/>  name: dinov2<br/>  size: base<br/>  frozen: true"]
        LC["language:<br/>  name: gpt2<br/>  frozen: true"]
        FC["fusion:<br/>  name: perceiver_resampler<br/>  num_latents: 64"]
        AC["action:<br/>  name: discrete_action<br/>  action_dim: 7"]
    end

    %% Factory Functions
    FACTORIES["Registry Factories"]

    subgraph factory_funcs["Factory Functions"]
        BV["build_vision_encoder()"]
        BL["build_language_encoder()"]
        BF["build_fusion_module()"]
        BA["build_action_head()"]
    end

    %% Instantiation
    subgraph instances["Component Instances"]
        VINST["DINOv2Backbone<br/>frozen=True"]
        LINST["GPT2Backbone<br/>frozen=True"]
        FINST["PerceiverResampler<br/>trainable"]
        AINST["DiscreteActionHead<br/>trainable"]
    end

    %% VLA Model
    VLA["VLA Model<br/>Assembled Pipeline"]

    %% Flow
    CONFIG --> VC
    CONFIG --> LC
    CONFIG --> FC
    CONFIG --> AC

    VC --> FACTORIES
    LC --> FACTORIES
    FC --> FACTORIES
    AC --> FACTORIES

    FACTORIES --> BV
    FACTORIES --> BL
    FACTORIES --> BF
    FACTORIES --> BA

    VREG --> BV
    LREG --> BL
    FREG --> BF
    AREG --> BA

    BV --> VINST
    BL --> LINST
    BF --> FINST
    BA --> AINST

    VINST --> VLA
    LINST --> VLA
    FINST --> VLA
    AINST --> VLA

    %% Styling
    classDef registry fill:#c8e6c9,stroke:#388e3c,stroke-width:2px
    classDef config fill:#fff9c4,stroke:#f9a825,stroke-width:2px
    classDef factory fill:#ffe0b2,stroke:#f57c00,stroke-width:2px
    classDef instance fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef model fill:#ffccbc,stroke:#d84315,stroke-width:3px

    class VREG,LREG,FREG,AREG registry
    class CONFIG,VC,LC,FC,AC config
    class FACTORIES,BV,BL,BF,BA factory
    class VINST,LINST,FINST,AINST instance
    class VLA model
```

---

## 7. Training Data Flow

Shows how gradients flow during training (frozen vs trainable components).

```mermaid
---
theme: default
---
flowchart TB
    %% Forward Pass
    subgraph forward["Forward Pass"]
        IMG["Images"]
        TXT["Text"]

        subgraph frozen_encoders["Frozen Encoders (No Gradients)"]
            VISION["Vision Encoder<br/>❄️ requires_grad=False"]
            LANG["Language Encoder<br/>❄️ requires_grad=False"]
        end

        VFEAT["Vision Features"]
        LFEAT["Language Features"]

        subgraph trainable["Trainable Modules (Gradients Flow)"]
            FUSION["🔥 Fusion Module<br/>requires_grad=True"]
            ACTION["🔥 Action Head<br/>requires_grad=True"]
        end

        FUSED["Fused Features"]
        PRED["Predicted Actions"]
    end

    %% Loss Computation
    GT["Ground Truth Actions"]
    LOSS["Loss Function<br/>CrossEntropy (discrete)<br/>or MSE (continuous)"]

    %% Backward Pass
    subgraph backward["Backward Pass (Gradients)"]
        GRAD_LOSS["∂L/∂pred"]
        GRAD_ACTION["∂L/∂action_head<br/>✓ Update weights"]
        GRAD_FUSION["∂L/∂fusion<br/>✓ Update weights"]
        GRAD_VISION["∂L/∂vision<br/>✗ Blocked (frozen)"]
        GRAD_LANG["∂L/∂language<br/>✗ Blocked (frozen)"]
    end

    %% Optimizer
    OPT["Optimizer<br/>Updates only trainable params"]

    %% Forward flow
    IMG --> VISION
    TXT --> LANG
    VISION --> VFEAT
    LANG --> LFEAT
    VFEAT --> FUSION
    LFEAT --> FUSION
    FUSION --> FUSED
    FUSED --> ACTION
    ACTION --> PRED
    PRED --> LOSS
    GT --> LOSS

    %% Backward flow
    LOSS -.backward().- GRAD_LOSS
    GRAD_LOSS -.-> GRAD_ACTION
    GRAD_ACTION -.-> GRAD_FUSION
    GRAD_FUSION -."✗ stopped".-> GRAD_VISION
    GRAD_FUSION -."✗ stopped".-> GRAD_LANG

    GRAD_ACTION --> OPT
    GRAD_FUSION --> OPT

    %% Styling
    classDef frozen fill:#bbdefb,stroke:#1976d2,stroke-width:3px,stroke-dasharray: 5 5
    classDef trainable fill:#ffccbc,stroke:#d84315,stroke-width:3px
    classDef data fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef loss fill:#fff59d,stroke:#f9a825,stroke-width:2px
    classDef gradient fill:#c8e6c9,stroke:#388e3c,stroke-width:2px,stroke-dasharray: 3 3

    class VISION,LANG frozen
    class FUSION,ACTION trainable
    class IMG,TXT,VFEAT,LFEAT,FUSED,PRED,GT data
    class LOSS,OPT loss
    class GRAD_LOSS,GRAD_ACTION,GRAD_FUSION,GRAD_VISION,GRAD_LANG gradient
```

---

## Summary Table

| Architecture | Vision | Language | Fusion | Action | Use Case |
|--------------|--------|----------|--------|--------|----------|
| **Arch 1** | DINOv2 | GPT-2 | Perceiver | Discrete | RT-2 style, robust binning |
| **Arch 2** | SigLIP | GPT-2 | Cross-Attn | Gaussian | Continuous control, uncertainty |
| **Arch 3** | Generic ViT | BERT | Gated | Hybrid | Mixed discrete/continuous |
| **Arch 4** | DINOv2 | GPT-2 | Temporal Perceiver | Discrete | Video/multi-frame tasks |

---

## Key Design Principles

### 🔒 Frozen Components (Transfer Learning)
- **Vision Encoders**: Pretrained on ImageNet/vision tasks, weights frozen
- **Language Encoders**: Pretrained on text corpora, weights frozen
- **Why**: Leverage strong pretrained representations without catastrophic forgetting

### 🔥 Trainable Components (Task-Specific)
- **Fusion Modules**: Learn how to combine vision + language for robotics
- **Action Heads**: Learn action space mapping specific to robot/task
- **Why**: Adapt multimodal understanding to action prediction

### 📊 Fixed-Size Bottleneck
- Fusion modules compress variable-length inputs (196 vision patches + variable text length) into fixed-size latents (e.g., 64 tokens)
- Enables consistent downstream processing regardless of input size
- Perceiver architecture is particularly effective for this compression

### 🎯 Action Representation
- **Discrete (Binned)**: More stable training, used in RT-2/OpenVLA, 256 bins per dimension
- **Gaussian (Continuous)**: Enables stochastic policies, uncertainty quantification
- **Hybrid**: Best of both worlds for mixed action spaces

---

## Notes

- All diagrams use the default batch size `B` and common dimensions from the codebase
- Vision features typically have `N=196` patches (14×14 grid for 224×224 images with patch size 16)
- Language features have variable length `L` (up to 77 tokens for GPT-2 in this implementation)
- Fusion outputs are typically 64 latent tokens of dimension 768
- Action dimension is 7 for standard robotic arms (6 DOF + gripper)

Generated with Mermaid.js v11 syntax for tinyVLA project documentation.
