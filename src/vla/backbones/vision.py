"""Vision backbone wrapper for timm models."""

from typing import Literal, Optional

import timm
import torch
import torch.nn as nn

from vla.registry import VISION_REGISTRY
from vla.utils import setup_logger

logger = setup_logger(__name__)


@VISION_REGISTRY.register("timm_vit")
class VisionBackbone(nn.Module):
    """Wrapper around timm vision models for VLA.

    Provides a unified interface for any timm vision model with VLA-specific
    features like output mode selection and optional projection layers.

    Args:
        model_name: timm model name (e.g., "vit_base_patch16_224")
        pretrained: Load pretrained weights
        frozen: Freeze backbone parameters (recommended for transfer learning)
        output_mode: Feature extraction mode:
            - "cls": Only CLS token [B, 1, D]
            - "spatial": Only patch tokens [B, N, D]
            - "both": CLS + patch tokens [B, N+1, D]
        proj_dim: Optional projection dimension for adapting feature size

    Example:
        >>> backbone = VisionBackbone(
        ...     model_name="vit_base_patch16_224",
        ...     pretrained=True,
        ...     frozen=True,
        ...     output_mode="spatial"
        ... )
        >>> images = torch.randn(2, 3, 224, 224)
        >>> features = backbone(images)
        >>> print(features.shape)
        torch.Size([2, 196, 768])
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        frozen: bool = True,
        output_mode: Literal["cls", "spatial", "both"] = "spatial",
        proj_dim: Optional[int] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.output_mode = output_mode

        logger.info(
            f"Loading timm model: {model_name} " f"(pretrained={pretrained}, frozen={frozen})"
        )

        # Load timm model without classification head
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
        )
        self.embed_dim: int = self.backbone.embed_dim  # type: ignore[assignment]

        # Freeze if requested
        if frozen:
            self._freeze_backbone()
            logger.info(f"Froze {model_name} backbone parameters")

        # Optional projection layer
        self.proj: Optional[nn.Linear] = None
        if proj_dim is not None:
            self.proj = nn.Linear(self.embed_dim, proj_dim)
            self.embed_dim = proj_dim
            logger.info(f"Added projection layer: {self.backbone.embed_dim} -> {proj_dim}")

    def _freeze_backbone(self) -> None:
        """Freeze all backbone parameters to prevent gradient updates."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through vision backbone.

        Args:
            x: Image tensor [B, C, H, W] normalized to [0, 1]

        Returns:
            features: Feature tensor [B, N, D] where N depends on output_mode:
                - cls: N=1 (CLS token only)
                - spatial: N=num_patches (patch tokens only)
                - both: N=num_patches+1 (CLS + patch tokens)

        Raises:
            ValueError: If image shape is invalid
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D image tensor [B, C, H, W], got shape {x.shape}")

        # Get all tokens including CLS via timm's forward_features
        features: torch.Tensor = self.backbone.forward_features(x)  # type: ignore[assignment,operator]

        if self.output_mode == "cls":
            # Extract just CLS token [B, D]
            if hasattr(self.backbone, "fc_norm"):
                # Apply final norm for models that use it
                cls_features: torch.Tensor = self.backbone.fc_norm(features[:, 0])  # type: ignore[assignment,operator]
            else:
                cls_features = features[:, 0]
            features = cls_features.unsqueeze(1)  # [B, 1, D]

        elif self.output_mode == "spatial":
            # Patch tokens only (exclude CLS) [B, N, D]
            features = features[:, 1:]

        # else "both": keep all tokens [B, N+1, D]

        # Apply optional projection
        if self.proj is not None:
            features = self.proj(features)

        return features

    @property
    def num_patches(self) -> int:
        """Number of spatial patches (excluding CLS token).

        Returns:
            Number of patches for the current input resolution
        """
        return self.backbone.patch_embed.num_patches  # type: ignore[return-value,union-attr]


@VISION_REGISTRY.register("dinov2")
class DINOv2Backbone(VisionBackbone):
    """DINOv2 backbone with sensible defaults.

    DINOv2 is a self-supervised vision transformer trained on large-scale data.
    It provides strong visual representations without requiring labeled data.

    Args:
        size: Model size variant
        pretrained: Load pretrained weights
        frozen: Freeze backbone parameters
        proj_dim: Optional projection dimension

    Example:
        >>> backbone = DINOv2Backbone(size="base", pretrained=True)
        >>> images = torch.randn(2, 3, 224, 224)
        >>> features = backbone(images)
        >>> print(features.shape)
        torch.Size([2, 196, 768])
    """

    def __init__(
        self,
        size: Literal["small", "base", "large", "giant"] = "base",
        pretrained: bool = True,
        frozen: bool = True,
        proj_dim: Optional[int] = None,
    ):
        model_map = {
            "small": "vit_small_patch14_dinov2.lvd142m",
            "base": "vit_base_patch14_dinov2.lvd142m",
            "large": "vit_large_patch14_dinov2.lvd142m",
            "giant": "vit_giant_patch14_dinov2.lvd142m",
        }
        super().__init__(
            model_name=model_map[size],
            pretrained=pretrained,
            frozen=frozen,
            output_mode="spatial",  # DINOv2 typically uses spatial features
            proj_dim=proj_dim,
        )


@VISION_REGISTRY.register("siglip")
class SigLIPBackbone(VisionBackbone):
    """SigLIP backbone for vision-language alignment.

    SigLIP (Sigmoid Loss for Language-Image Pre-training) is designed for
    vision-language tasks with improved efficiency over CLIP.

    Args:
        size: Model size variant
        pretrained: Load pretrained weights
        frozen: Freeze backbone parameters
        proj_dim: Optional projection dimension

    Example:
        >>> backbone = SigLIPBackbone(size="base", pretrained=True)
        >>> images = torch.randn(2, 3, 224, 224)
        >>> features = backbone(images)
        >>> print(features.shape)
        torch.Size([2, 196, 768])
    """

    def __init__(
        self,
        size: Literal["base", "large"] = "base",
        pretrained: bool = True,
        frozen: bool = True,
        proj_dim: Optional[int] = None,
    ):
        model_map = {
            "base": "vit_base_patch16_siglip_224",
            "large": "vit_large_patch16_siglip_256",
        }
        super().__init__(
            model_name=model_map[size],
            pretrained=pretrained,
            frozen=frozen,
            output_mode="spatial",  # SigLIP uses spatial features
            proj_dim=proj_dim,
        )
