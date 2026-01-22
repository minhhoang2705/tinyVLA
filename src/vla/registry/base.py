"""Generic registry for component registration and lookup."""

from typing import Any, Callable, Dict, Generic, Type, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Type-safe registry for component classes.

    Provides decorator-based registration and dynamic instantiation of components.
    Supports O(1) lookup and integrates seamlessly with Hydra configuration.

    Usage:
        >>> vision_registry = Registry[nn.Module]("vision")
        >>> @vision_registry.register("vit_base")
        ... class ViTBase(nn.Module):
        ...     def __init__(self, hidden_dim: int = 768):
        ...         super().__init__()
        ...         self.hidden_dim = hidden_dim
        >>> model = vision_registry.get("vit_base", hidden_dim=1024)

    Args:
        name: Human-readable name for this registry (used in error messages)
    """

    def __init__(self, name: str):
        """Initialize registry with a name.

        Args:
            name: Identifier for this registry (e.g., "vision", "language")
        """
        self._name = name
        self._registry: Dict[str, Type[T]] = {}

    def register(self, name: str) -> Callable[[Type[T]], Type[T]]:
        """Decorator to register a class under a string name.

        Args:
            name: String identifier for the component

        Returns:
            Decorator function that registers the class

        Raises:
            ValueError: If name is already registered

        Example:
            >>> @registry.register("my_component")
            ... class MyComponent:
            ...     pass
        """

        def wrapper(cls: Type[T]) -> Type[T]:
            if name in self._registry:
                raise ValueError(
                    f"Component '{name}' already registered in {self._name} registry"
                )
            self._registry[name] = cls
            return cls

        return wrapper

    def get(self, name: str, **kwargs: Any) -> T:
        """Instantiate a registered class with kwargs.

        Args:
            name: Registered component name
            **kwargs: Arguments passed to component constructor

        Returns:
            Instance of the registered component

        Raises:
            KeyError: If name is not registered

        Example:
            >>> module = registry.get("vit_base", hidden_dim=512)
        """
        if name not in self._registry:
            available = ", ".join(self._registry.keys()) if self._registry else "none"
            raise KeyError(
                f"Component '{name}' not found in {self._name} registry. "
                f"Available components: {available}"
            )
        return self._registry[name](**kwargs)

    def get_class(self, name: str) -> Type[T]:
        """Get the registered class without instantiating it.

        Args:
            name: Registered component name

        Returns:
            The registered class (not an instance)

        Raises:
            KeyError: If name is not registered

        Example:
            >>> cls = registry.get_class("vit_base")
            >>> instance = cls(hidden_dim=768)
        """
        if name not in self._registry:
            available = ", ".join(self._registry.keys()) if self._registry else "none"
            raise KeyError(
                f"Component '{name}' not found in {self._name} registry. "
                f"Available components: {available}"
            )
        return self._registry[name]

    def list_available(self) -> list[str]:
        """List all registered component names.

        Returns:
            Sorted list of registered component names

        Example:
            >>> registry.list_available()
            ['resnet50', 'vit_base', 'vit_large']
        """
        return sorted(self._registry.keys())

    def __contains__(self, name: str) -> bool:
        """Check if a component name is registered.

        Args:
            name: Component name to check

        Returns:
            True if the component is registered, False otherwise

        Example:
            >>> "vit_base" in registry
            True
        """
        return name in self._registry

    def __repr__(self) -> str:
        """String representation of the registry.

        Returns:
            Human-readable description
        """
        count = len(self._registry)
        return f"Registry(name='{self._name}', components={count})"


# Global registries for each component type
VISION_REGISTRY = Registry[Any]("vision")
LANGUAGE_REGISTRY = Registry[Any]("language")
FUSION_REGISTRY = Registry[Any]("fusion")
ACTION_REGISTRY = Registry[Any]("action")
MODEL_REGISTRY = Registry[Any]("model")
