"""
Data transformation utilities for PASCAL VOC dataset.
"""
from typing import List, Dict, Tuple, Optional, Union, Any
import random

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np


class SimpleTransform:
    """
    Simple transformation for cases when bounding box adjustment is not needed.
    """

    def __init__(
            self,
            img_size: int = 224,
            is_train: bool = True,
            mean: List[float] = [0.485, 0.456, 0.406],
            std: List[float] = [0.229, 0.224, 0.225]
    ):
        """
        Initialize simple transformations.

        Args:
            img_size: Size to which images will be resized.
            is_train: Whether this is for training or validation.
            mean: Mean values for normalization.
            std: Standard deviation values for normalization.
        """
        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

    def __call__(self, img: Image.Image) -> torch.Tensor:
        """
        Apply transformations to image.

        Args:
            img: PIL Image to transform.

        Returns:
            Transformed image tensor.
        """
        return self.transform(img)


def create_simple_transform(
        img_size: int = 224,
        is_train: bool = True,
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225]
) -> SimpleTransform:
    """
    Create a simple transformation function.

    Args:
        img_size: Size to which images will be resized.
        is_train: Whether this is for training or validation.
        mean: Mean values for normalization.
        std: Standard deviation values for normalization.

    Returns:
        SimpleTransform object.
    """
    return SimpleTransform(
        img_size=img_size,
        is_train=is_train,
        mean=mean,
        std=std
    )

