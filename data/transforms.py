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


class GRAFTTransform:
    """
    Custom transformations for GRAFT model with bounding box handling.

    This class handles transformations for both images and their corresponding
    bounding boxes, ensuring that the boxes are properly adjusted after
    image transformations.
    """

    def __init__(
            self,
            img_size: int = 224,
            is_train: bool = True,
            mean: List[float] = [0.485, 0.456, 0.406],
            std: List[float] = [0.229, 0.224, 0.225],
            random_horizontal_flip: bool = True,
            random_crop: bool = True,
            color_jitter: bool = True
    ):
        """
        Initialize transformations.

        Args:
            img_size: Size to which images will be resized.
            is_train: Whether this is for training or validation.
            mean: Mean values for normalization.
            std: Standard deviation values for normalization.
            random_horizontal_flip: Whether to apply random horizontal flip.
            random_crop: Whether to apply random crop.
            color_jitter: Whether to apply color jitter.
        """
        self.img_size = img_size
        self.is_train = is_train
        self.mean = mean
        self.std = std
        self.random_horizontal_flip = random_horizontal_flip and is_train
        self.random_crop = random_crop and is_train
        self.color_jitter = color_jitter and is_train

        # Define jitter parameters for training
        self.brightness = 0.2
        self.contrast = 0.2
        self.saturation = 0.2
        self.hue = 0.1

        # Random crop parameters
        self.crop_scale = (0.5, 1.0)
        self.crop_ratio = (0.75, 1.33)

        # Flip probability
        self.flip_prob = 0.5

    def __call__(self, img: Image.Image, boxes: Optional[List[List[float]]] = None) -> Tuple[
        torch.Tensor, Optional[List[List[float]]]]:
        """
        Apply transformations to image and adjust bounding boxes accordingly.

        Args:
            img: PIL Image to transform.
            boxes: List of bounding boxes [x_min, y_min, x_max, y_max].

        Returns:
            Tuple containing:
                - Transformed image tensor
                - Adjusted bounding boxes
        """
        width, height = img.size

        # Keep track of original image size for box normalization
        orig_width, orig_height = width, height

        # Apply transformations and adjust boxes
        if self.is_train:
            # Random crop
            if self.random_crop:
                i, j, h, w = transforms.RandomResizedCrop.get_params(
                    img, scale=self.crop_scale, ratio=self.crop_ratio
                )
                img = F.crop(img, i, j, h, w)

                # Adjust bounding boxes for crop
                if boxes is not None:
                    adjusted_boxes = []
                    for box in boxes:
                        x_min, y_min, x_max, y_max = box

                        # Convert to crop coordinates
                        x_min = max(0, (x_min - j) / w * width)
                        y_min = max(0, (y_min - i) / h * height)
                        x_max = min(width, (x_max - j) / w * width)
                        y_max = min(height, (y_max - i) / h * height)

                        # Skip invalid boxes
                        if x_min >= x_max or y_min >= y_max:
                            continue

                        adjusted_boxes.append([x_min, y_min, x_max, y_max])

                    boxes = adjusted_boxes

            # Color jitter
            if self.color_jitter:
                img = F.adjust_brightness(img, random.uniform(1 - self.brightness, 1 + self.brightness))
                img = F.adjust_contrast(img, random.uniform(1 - self.contrast, 1 + self.contrast))
                img = F.adjust_saturation(img, random.uniform(1 - self.saturation, 1 + self.saturation))
                img = F.adjust_hue(img, random.uniform(-self.hue, self.hue))

            # Random horizontal flip
            if self.random_horizontal_flip and random.random() < self.flip_prob:
                img = F.hflip(img)

                # Adjust bounding boxes for horizontal flip
                if boxes is not None:
                    adjusted_boxes = []
                    for box in boxes:
                        x_min, y_min, x_max, y_max = box

                        # Flip x coordinates
                        flipped_x_min = width - x_max
                        flipped_x_max = width - x_min

                        adjusted_boxes.append([flipped_x_min, y_min, flipped_x_max, y_max])

                    boxes = adjusted_boxes

        # Resize to target size
        img = F.resize(img, (self.img_size, self.img_size))

        # Adjust bounding boxes for resize
        if boxes is not None:
            scale_x = self.img_size / width
            scale_y = self.img_size / height

            adjusted_boxes = []
            for box in boxes:
                x_min, y_min, x_max, y_max = box

                # Scale coordinates
                x_min = x_min * scale_x
                y_min = y_min * scale_y
                x_max = x_max * scale_x
                y_max = y_max * scale_y

                adjusted_boxes.append([x_min, y_min, x_max, y_max])

            boxes = adjusted_boxes

        # Convert to tensor
        img = F.to_tensor(img)

        # Normalize
        img = F.normalize(img, mean=self.mean, std=self.std)

        return img, boxes


def create_transform(
        img_size: int = 224,
        is_train: bool = True,
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        random_horizontal_flip: bool = True,
        random_crop: bool = True,
        color_jitter: bool = True
) -> GRAFTTransform:
    """
    Create a transformation function for the GRAFT model.

    Args:
        img_size: Size to which images will be resized.
        is_train: Whether this is for training or validation.
        mean: Mean values for normalization.
        std: Standard deviation values for normalization.
        random_horizontal_flip: Whether to apply random horizontal flip.
        random_crop: Whether to apply random crop.
        color_jitter: Whether to apply color jitter.

    Returns:
        GRAFTTransform object.
    """
    return GRAFTTransform(
        img_size=img_size,
        is_train=is_train,
        mean=mean,
        std=std,
        random_horizontal_flip=random_horizontal_flip,
        random_crop=random_crop,
        color_jitter=color_jitter
    )


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

