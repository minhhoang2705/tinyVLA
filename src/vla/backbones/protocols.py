"""Protocol interfaces for VLA backbone components.

Defines runtime-checkable Protocol interfaces that formalize the contracts
for vision encoders, language encoders, fusion modules, and action heads.

These protocols enable:
- Type checking at runtime via isinstance()
- Clear interface documentation
- Loose coupling between components
- Easy component swapping

Example:
    >>> from vla.backbones.protocols import VisionBackboneProtocol
    >>> from vla.backbones import TimmVisionBackbone
    >>>
    >>> vision = TimmVisionBackbone("vit_tiny_patch16_224")
    >>> assert isinstance(vision, VisionBackboneProtocol)  # Runtime check
    >>> print(vision.embed_dim)  # Protocol guarantees this attribute
"""

from typing import List, Optional, Protocol, Tuple, runtime_checkable

import torch


@runtime_checkable
class VisionBackboneProtocol(Protocol):
    """Protocol for vision encoder modules.

    Vision backbones must:
    1. Expose an `embed_dim` attribute (int) indicating feature dimension
    2. Implement forward(images) returning features [B, N, D]

    Example:
        >>> class MyVisionEncoder:
        ...     def __init__(self):
        ...         self.embed_dim = 768
        ...
        ...     def forward(self, images: torch.Tensor) -> torch.Tensor:
        ...         # Extract features from images [B, 3, H, W]
        ...         return torch.randn(images.shape[0], 196, self.embed_dim)
        >>>
        >>> encoder = MyVisionEncoder()
        >>> assert isinstance(encoder, VisionBackboneProtocol)
    """

    embed_dim: int

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract visual features from images.

        Args:
            images: Input images [B, C, H, W]

        Returns:
            Visual features [B, N, D] where:
                - B is batch size
                - N is number of visual tokens (e.g., 196 for ViT 14x14 patches)
                - D is embed_dim (feature dimension)

        Example:
            >>> images = torch.randn(4, 3, 224, 224)
            >>> features = vision.forward(images)
            >>> features.shape
            torch.Size([4, 196, 768])
        """
        ...


@runtime_checkable
class LanguageBackboneProtocol(Protocol):
    """Protocol for language encoder modules.

    Language backbones must:
    1. Expose an `embed_dim` attribute (int) indicating feature dimension
    2. Implement forward() accepting texts OR input_ids
    3. Return language features [B, L, D] or [B, D] depending on output_mode

    Example:
        >>> class MyLanguageEncoder:
        ...     def __init__(self):
        ...         self.embed_dim = 768
        ...
        ...     def forward(
        ...         self,
        ...         texts: Optional[List[str]] = None,
        ...         input_ids: Optional[torch.Tensor] = None,
        ...         attention_mask: Optional[torch.Tensor] = None,
        ...     ) -> torch.Tensor:
        ...         batch_size = len(texts) if texts else input_ids.shape[0]
        ...         return torch.randn(batch_size, 10, self.embed_dim)
        >>>
        >>> encoder = MyLanguageEncoder()
        >>> assert isinstance(encoder, LanguageBackboneProtocol)
    """

    embed_dim: int

    def forward(
        self,
        texts: Optional[List[str]] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode text instructions into features.

        Args:
            texts: List of instruction strings [B]. Mutually exclusive with input_ids.
            input_ids: Pre-tokenized input IDs [B, L]. Mutually exclusive with texts.
            attention_mask: Attention mask [B, L] for input_ids

        Returns:
            Language features [B, L, D] or [B, D] where:
                - B is batch size
                - L is sequence length (variable, depends on tokenization)
                - D is embed_dim (feature dimension)

        Note:
            Exactly one of texts or input_ids must be provided.

        Example:
            >>> texts = ["pick up cube", "place on table"]
            >>> features = language.forward(texts=texts)
            >>> features.shape
            torch.Size([2, 10, 768])
        """
        ...


@runtime_checkable
class FusionModuleProtocol(Protocol):
    """Protocol for multimodal fusion modules.

    Fusion modules must:
    1. Implement forward() accepting vision and language features
    2. Return fused features with fixed-size latent dimension [B, K, D]

    The fusion module compresses variable-length vision/language inputs
    into a fixed-size bottleneck representation.

    Example:
        >>> class MyFusionModule:
        ...     def forward(
        ...         self,
        ...         vision_features: torch.Tensor,
        ...         language_features: torch.Tensor,
        ...     ) -> torch.Tensor:
        ...         # Fuse modalities into fixed 64 latent tokens
        ...         batch_size = vision_features.shape[0]
        ...         return torch.randn(batch_size, 64, 768)
        >>>
        >>> fusion = MyFusionModule()
        >>> assert isinstance(fusion, FusionModuleProtocol)
    """

    def forward(
        self,
        vision_features: torch.Tensor,
        language_features: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse vision and language features into unified representation.

        Args:
            vision_features: Visual features [B, N, D_v]
            language_features: Language features [B, L, D_l] or [B, D_l]

        Returns:
            Fused features [B, K, D] where:
                - B is batch size
                - K is fixed number of latent tokens (e.g., 64)
                - D is fusion output dimension

        Note:
            The fusion module MUST compress to fixed K latents regardless
            of input sequence lengths N and L. This enables consistent
            downstream processing.

        Example:
            >>> vision = torch.randn(4, 196, 768)
            >>> language = torch.randn(4, 20, 768)
            >>> fused = fusion.forward(vision, language)
            >>> fused.shape
            torch.Size([4, 64, 768])
        """
        ...


@runtime_checkable
class ActionHeadProtocol(Protocol):
    """Protocol for action prediction heads.

    Action heads must:
    1. Implement forward() accepting fused features
    2. Support return_logits parameter
    3. Return actions [B, action_dim] and optionally logits
    4. Implement compute_loss() for training

    Example:
        >>> class MyActionHead:
        ...     def forward(
        ...         self,
        ...         fused_features: torch.Tensor,
        ...         return_logits: bool = False,
        ...     ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        ...         batch_size = fused_features.shape[0]
        ...         actions = torch.randn(batch_size, 7)
        ...         logits = torch.randn(batch_size, 7, 256) if return_logits else None
        ...         return actions, logits
        ...
        ...     def compute_loss(
        ...         self,
        ...         logits: torch.Tensor,
        ...         target_actions: torch.Tensor,
        ...     ) -> torch.Tensor:
        ...         return torch.nn.functional.mse_loss(logits.mean(dim=-1), target_actions)
        >>>
        >>> head = MyActionHead()
        >>> assert isinstance(head, ActionHeadProtocol)
    """

    def forward(
        self,
        fused_features: torch.Tensor,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Predict actions from fused features.

        Args:
            fused_features: Fused multimodal features [B, K, D]
            return_logits: Whether to return raw logits for loss computation

        Returns:
            Tuple of (actions, logits) where:
                - actions: Predicted actions [B, action_dim] in range [-1, 1]
                - logits: Raw logits for loss computation (optional)
                  - For discrete: [B, action_dim, num_bins]
                  - For Gaussian: (mean, std) each [B, action_dim]

        Example:
            >>> fused = torch.randn(4, 64, 768)
            >>> actions, logits = head.forward(fused, return_logits=True)
            >>> actions.shape
            torch.Size([4, 7])
            >>> logits.shape  # Discrete case
            torch.Size([4, 7, 256])
        """
        ...

    def compute_loss(
        self,
        logits: torch.Tensor,
        target_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute action prediction loss.

        Args:
            logits: Raw logits from forward pass
                - For discrete: [B, action_dim, num_bins]
                - For Gaussian: mean [B, action_dim] (or tuple with std)
            target_actions: Ground truth actions [B, action_dim] in range [-1, 1]

        Returns:
            Scalar loss tensor

        Example:
            >>> logits = torch.randn(4, 7, 256)  # Discrete
            >>> target_actions = torch.randn(4, 7).clamp(-1, 1)
            >>> loss = head.compute_loss(logits, target_actions)
            >>> loss.shape
            torch.Size([])
        """
        ...


def validate_vision_backbone(module: object) -> bool:
    """Check if module implements VisionBackboneProtocol.

    Args:
        module: Module to validate

    Returns:
        True if module implements protocol, False otherwise

    Example:
        >>> from vla.backbones import TimmVisionBackbone
        >>> vision = TimmVisionBackbone("vit_tiny_patch16_224")
        >>> validate_vision_backbone(vision)
        True
    """
    return isinstance(module, VisionBackboneProtocol)


def validate_language_backbone(module: object) -> bool:
    """Check if module implements LanguageBackboneProtocol.

    Args:
        module: Module to validate

    Returns:
        True if module implements protocol, False otherwise
    """
    return isinstance(module, LanguageBackboneProtocol)


def validate_fusion_module(module: object) -> bool:
    """Check if module implements FusionModuleProtocol.

    Args:
        module: Module to validate

    Returns:
        True if module implements protocol, False otherwise
    """
    return isinstance(module, FusionModuleProtocol)


def validate_action_head(module: object) -> bool:
    """Check if module implements ActionHeadProtocol.

    Args:
        module: Module to validate

    Returns:
        True if module implements protocol, False otherwise
    """
    return isinstance(module, ActionHeadProtocol)
