"""Multi-scale feature extraction from vision backbones."""

from typing import Dict, List, Optional

import timm
import torch
import torch.nn as nn

from vla.utils import setup_logger

logger = setup_logger(__name__)


class MultiScaleFeatureExtractor(nn.Module):
    """Extract features from multiple backbone layers.

    Useful for dense prediction tasks or multi-resolution fusion where
    features from different depths of the network are beneficial.

    Args:
        model_name: timm model name
        pretrained: Load pretrained weights
        frozen: Freeze backbone parameters
        layers: Layer indices to extract features from (negative indexing supported)

    Example:
        >>> extractor = MultiScaleFeatureExtractor(
        ...     model_name="vit_base_patch16_224",
        ...     layers=[-4, -3, -2, -1]  # Last 4 layers
        ... )
        >>> images = torch.randn(2, 3, 224, 224)
        >>> features = extractor(images)
        >>> print({k: v.shape for k, v in features.items()})
        {8: torch.Size([2, 197, 768]), 9: torch.Size([2, 197, 768]), ...}
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        frozen: bool = True,
        layers: Optional[List[int]] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.layers = layers if layers is not None else [-4, -3, -2, -1]

        logger.info(f"Loading multi-scale extractor for {model_name} at layers {layers}")

        # Load timm model
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=False,  # We'll use hooks instead
        )

        # Freeze if requested
        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info(f"Froze {model_name} backbone parameters")

        # Hook storage
        self._features: Dict[int, torch.Tensor] = {}
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward hooks on target layers."""
        if not hasattr(self.backbone, "blocks"):
            raise ValueError(
                f"Model {self.model_name} does not have 'blocks' attribute. "
                "Multi-scale extraction only supports ViT-like models."
            )

        blocks = self.backbone.blocks  # type: ignore[attr-defined,unused-ignore]
        num_blocks: int = len(blocks)  # type: ignore[arg-type,unused-ignore]

        for idx in self.layers:
            # Convert negative indexing to positive
            layer_idx = idx if idx >= 0 else num_blocks + idx

            if layer_idx < 0 or layer_idx >= num_blocks:
                raise ValueError(
                    f"Invalid layer index {idx} (resolved to {layer_idx}). "
                    f"Model has {num_blocks} blocks."
                )

            blocks[layer_idx].register_forward_hook(self._make_hook(layer_idx))  # type: ignore[index,union-attr]
            logger.debug(f"Registered hook at layer {layer_idx}")

    def _make_hook(self, layer_idx: int) -> object:
        """Create a forward hook that stores layer output.

        Args:
            layer_idx: Index of the layer to hook

        Returns:
            Hook function
        """

        def hook(
            module: nn.Module, input: tuple[torch.Tensor, ...], output: torch.Tensor
        ) -> None:
            self._features[layer_idx] = output

        return hook

    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Extract multi-scale features.

        Args:
            x: Image tensor [B, C, H, W]

        Returns:
            Dictionary mapping layer index to features [B, N, D]

        Raises:
            ValueError: If image shape is invalid
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D image tensor [B, C, H, W], got shape {x.shape}")

        # Clear previous features
        self._features.clear()

        # Forward pass (hooks will populate self._features)
        _: torch.Tensor = self.backbone.forward_features(x)  # type: ignore[assignment,operator,unused-ignore]

        # Return copy of features dict
        return dict(self._features)


class DualEncoderVision(nn.Module):
    """Dual vision encoder (e.g., DINOv2 + SigLIP) following OpenVLA.

    Concatenates features from two complementary vision backbones to combine
    their strengths. For example, DINOv2 provides strong geometric features
    while SigLIP offers vision-language alignment.

    Args:
        encoder1_name: First encoder's timm model name
        encoder2_name: Second encoder's timm model name
        pretrained: Load pretrained weights for both encoders
        frozen: Freeze both backbone parameters
        proj_dim: Optional projection dimension for combined features

    Example:
        >>> encoder = DualEncoderVision(
        ...     encoder1_name="vit_base_patch14_dinov2.lvd142m",
        ...     encoder2_name="vit_base_patch16_siglip_224",
        ...     proj_dim=768
        ... )
        >>> images = torch.randn(2, 3, 224, 224)
        >>> features = encoder(images)
        >>> print(features.shape)
        torch.Size([2, 196, 768])
    """

    def __init__(
        self,
        encoder1_name: str = "vit_base_patch14_dinov2.lvd142m",
        encoder2_name: str = "vit_base_patch16_siglip_224",
        pretrained: bool = True,
        frozen: bool = True,
        proj_dim: Optional[int] = None,
    ):
        super().__init__()

        # Import here to avoid circular imports
        from .vision import VisionBackbone

        logger.info(f"Creating dual encoder: {encoder1_name} + {encoder2_name}")

        # Create both encoders
        self.encoder1 = VisionBackbone(
            model_name=encoder1_name,
            pretrained=pretrained,
            frozen=frozen,
            output_mode="spatial",
        )
        self.encoder2 = VisionBackbone(
            model_name=encoder2_name,
            pretrained=pretrained,
            frozen=frozen,
            output_mode="spatial",
        )

        # Calculate combined dimension
        combined_dim: int = self.encoder1.embed_dim + self.encoder2.embed_dim  # type: ignore[assignment,unused-ignore]
        logger.info(
            f"Combined feature dim: {self.encoder1.embed_dim} + "
            f"{self.encoder2.embed_dim} = {combined_dim}"
        )

        # Optional projection to reduce dimension
        self.proj: Optional[nn.Linear] = None
        self.embed_dim: int
        if proj_dim is not None:
            self.proj = nn.Linear(combined_dim, proj_dim)
            self.embed_dim = proj_dim
            logger.info(f"Adding projection: {combined_dim} -> {proj_dim}")
        else:
            self.embed_dim = combined_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through dual encoders.

        Args:
            x: Image tensor [B, C, H, W]

        Returns:
            Combined features [B, N, D] where N is the number of patches
            and D is either combined_dim or proj_dim

        Raises:
            ValueError: If image shape is invalid
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D image tensor [B, C, H, W], got shape {x.shape}")

        # Extract features from both encoders
        feat1 = self.encoder1(x)  # [B, N1, D1]
        feat2 = self.encoder2(x)  # [B, N2, D2]

        # Handle different patch counts via interpolation
        if feat1.size(1) != feat2.size(1):
            logger.debug(f"Interpolating features: {feat2.size(1)} -> {feat1.size(1)} patches")
            # Interpolate feat2 to match feat1's sequence length
            # [B, N2, D2] -> [B, D2, N2] -> interpolate -> [B, D2, N1] -> [B, N1, D2]
            feat2 = nn.functional.interpolate(
                feat2.transpose(1, 2),
                size=feat1.size(1),
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)

        # Concatenate along feature dimension
        combined = torch.cat([feat1, feat2], dim=-1)  # [B, N, D1+D2]

        # Apply optional projection
        if self.proj is not None:
            combined = self.proj(combined)  # [B, N, proj_dim]

        return combined
