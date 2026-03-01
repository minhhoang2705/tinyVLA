from typing import Dict, Any, Callable
from torch.utils.data import Dataset

class TransformWrapper(Dataset):
    """Wraps a PyTorch dataset to apply a transformation to the returned samples.
    
    This is especially useful when using random_split() where you want to 
    apply data augmentation to the training split but NOT the validation split.
    
    Args:
        dataset: The underlying PyTorch Dataset (or Subset)
        transform: A callable that takes a sample dict and returns a modified sample dict
    """
    def __init__(self, dataset: Dataset, transform: Callable):
        self.dataset = dataset
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.dataset)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.dataset[idx]
        return self.transform(sample)
