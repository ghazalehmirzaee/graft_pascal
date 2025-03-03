"""
PASCAL VOC dataset implementation for GRAFT framework.
"""
import os
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple, Optional, Callable, Union

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torchvision.transforms as transforms

# PASCAL VOC class names (20 classes)
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# Global variable to store class counts
_class_counts = None


class PascalVOCDataset(Dataset):
    """
    PASCAL VOC dataset for multi-label classification.

    This implementation handles both the training and validation splits
    and provides bounding box information for graph construction.
    """

    def __init__(
            self,
            root: str,
            year: str = "2012",
            split: str = "train",
            transform: Optional[Callable] = None,
            download: bool = False,
            keep_difficult: bool = False,
            return_boxes: bool = True
    ):
        """
        Initialize PASCAL VOC dataset.

        Args:
            root: Root directory of the PASCAL VOC Dataset.
            year: Dataset year, either "2012" or "2007".
            split: Dataset split, either "train", "val", or "trainval".
            transform: Image transformation function.
            download: Whether to download the dataset if not present.
            keep_difficult: Whether to include difficult examples.
            return_boxes: Whether to return bounding box information.
        """
        self.root = root
        self.year = year
        self.split = split
        self.transform = transform
        self.keep_difficult = keep_difficult
        self.return_boxes = return_boxes

        self.image_dir = os.path.join(root, f"VOC{year}", "JPEGImages")
        self.anno_dir = os.path.join(root, f"VOC{year}", "Annotations")
        self.split_dir = os.path.join(root, f"VOC{year}", "ImageSets", "Main")

        # If download is True and the dataset doesn't exist, download it
        if download and not os.path.exists(self.image_dir):
            self._download()

        # Read image IDs from the specified split
        self.ids = self._read_image_ids()

        # Compute class weights for handling class imbalance
        self.class_weights = self._compute_class_weights()

        # Build class distribution
        self.class_counts = self._compute_class_counts()

        # Cache annotations for faster loading
        self.annotations_cache = {}

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.ids)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Get a sample from the dataset.

        Args:
            index: Index of the sample.

        Returns:
            Tuple containing:
                - image: Transformed image tensor
                - labels: Multi-hot encoded label tensor
                - metadata: Dictionary with additional information (bounding boxes, etc.)
        """
        img_id = self.ids[index]

        # Load image
        img_path = os.path.join(self.image_dir, f"{img_id}.jpg")
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder image and empty targets
            img = Image.new('RGB', (224, 224), color='gray')
            target = torch.zeros(len(VOC_CLASSES))
            metadata = {'image_id': img_id, 'boxes': [], 'labels': [], 'difficulties': []}
            if self.transform:
                img = self.transform(img)
            return img, target, metadata

        # Load annotations (from cache if available)
        if img_id in self.annotations_cache:
            boxes, labels, difficulties = self.annotations_cache[img_id]
        else:
            anno_path = os.path.join(self.anno_dir, f"{img_id}.xml")
            try:
                boxes, labels, difficulties = self._parse_annotation(anno_path)
                self.annotations_cache[img_id] = (boxes, labels, difficulties)
            except Exception as e:
                print(f"Error parsing annotation {anno_path}: {e}")
                boxes, labels, difficulties = [], [], []

        # Create multi-hot encoded label tensor
        target = torch.zeros(len(VOC_CLASSES))
        for label in labels:
            if difficulties[labels.index(label)] and not self.keep_difficult:
                continue
            target[label] = 1

        # Prepare metadata with bounding boxes and image info
        metadata = {
            'image_id': img_id,
        }

        if self.return_boxes:
            metadata.update({
                'boxes': boxes,
                'labels': labels,
                'difficulties': difficulties,
                'height': img.height,
                'width': img.width
            })

        # Apply transformations
        if self.transform:
            img = self.transform(img)

        return img, target, metadata

    def _read_image_ids(self) -> List[str]:
        """
        Read image IDs from the dataset split file.

        Returns:
            List of image IDs for the specified split.
        """
        split_file = os.path.join(self.split_dir, f"{self.split}.txt")
        with open(split_file, "r") as f:
            return [line.strip() for line in f.readlines()]

    def _parse_annotation(self, anno_path: str) -> Tuple[List[List[float]], List[int], List[bool]]:
        """
        Parse XML annotation file to extract bounding boxes and class labels.

        Args:
            anno_path: Path to the XML annotation file.

        Returns:
            Tuple containing:
                - boxes: List of bounding boxes [x_min, y_min, x_max, y_max]
                - labels: List of class indices
                - difficulties: List of difficulty flags
        """
        tree = ET.parse(anno_path)
        root = tree.getroot()

        boxes = []
        labels = []
        difficulties = []

        for obj in root.findall("object"):
            difficult = int(obj.find("difficult").text) == 1

            # Skip difficult instances if not keeping them
            if difficult and not self.keep_difficult:
                continue

            label_name = obj.find("name").text
            if label_name not in VOC_CLASSES:
                continue

            bbox = obj.find("bndbox")
            x_min = float(bbox.find("xmin").text)
            y_min = float(bbox.find("ymin").text)
            x_max = float(bbox.find("xmax").text)
            y_max = float(bbox.find("ymax").text)

            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(VOC_CLASSES.index(label_name))
            difficulties.append(difficult)

        return boxes, labels, difficulties

    def _compute_class_weights(self) -> torch.Tensor:
        """
        Compute class weights based on class frequency.

        Returns:
            Tensor of class weights.
        """
        class_counts = torch.zeros(len(VOC_CLASSES))

        # Count instances of each class
        for img_id in self.ids:
            anno_path = os.path.join(self.anno_dir, f"{img_id}.xml")
            if not os.path.exists(anno_path):
                continue

            try:
                tree = ET.parse(anno_path)
                root = tree.getroot()

                for obj in root.findall("object"):
                    if int(obj.find("difficult").text) == 1 and not self.keep_difficult:
                        continue

                    label_name = obj.find("name").text
                    if label_name in VOC_CLASSES:
                        class_counts[VOC_CLASSES.index(label_name)] += 1
            except Exception as e:
                print(f"Error computing class weights for {anno_path}: {e}")

        # Compute weights as inverse of frequency with smoothing
        smooth_factor = 1.0
        class_weights = 1.0 / torch.max(torch.sqrt(class_counts), torch.ones_like(class_counts) * smooth_factor)

        # Normalize weights
        class_weights = class_weights / torch.sum(class_weights) * len(VOC_CLASSES)

        return class_weights

    def _compute_class_counts(self) -> List[int]:
        """
        Compute the number of samples for each class.

        Returns:
            List of class counts.
        """
        class_counts = [0] * len(VOC_CLASSES)

        # Count instances of each class
        for img_id in self.ids:
            anno_path = os.path.join(self.anno_dir, f"{img_id}.xml")
            if not os.path.exists(anno_path):
                continue

            try:
                tree = ET.parse(anno_path)
                root = tree.getroot()

                for obj in root.findall("object"):
                    if int(obj.find("difficult").text) == 1 and not self.keep_difficult:
                        continue

                    label_name = obj.find("name").text
                    if label_name in VOC_CLASSES:
                        class_counts[VOC_CLASSES.index(label_name)] += 1
            except Exception as e:
                print(f"Error computing class counts for {anno_path}: {e}")

        # Update global class counts
        global _class_counts
        _class_counts = class_counts

        return class_counts

    def _download(self):
        """
        Download and extract PASCAL VOC dataset.
        """
        raise NotImplementedError(
            "Automatic download not implemented. Please download the PASCAL VOC dataset manually."
        )


def create_pascal_voc_dataloaders(
        root: str,
        img_size: int = 224,
        batch_size: int = 32,
        num_workers: int = 8,
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        distributed: bool = False,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        prefetch_factor: int = 2,
        drop_last: bool = True
) -> Tuple[DataLoader, DataLoader, PascalVOCDataset, PascalVOCDataset]:
    """
    Create train and validation dataloaders for PASCAL VOC.

    Args:
        root: Root directory of the PASCAL VOC dataset.
        img_size: Size to which images will be resized.
        batch_size: Batch size for dataloaders.
        num_workers: Number of workers for dataloaders.
        mean: Mean values for normalization.
        std: Standard deviation values for normalization.
        distributed: Whether to use distributed training.
        pin_memory: Whether to pin memory in DataLoader.
        persistent_workers: Whether to use persistent workers.
        prefetch_factor: Prefetch factor for DataLoader.
        drop_last: Whether to drop the last incomplete batch.

    Returns:
        Tuple containing:
            - train_loader: DataLoader for training set
            - val_loader: DataLoader for validation set
            - train_dataset: Training dataset
            - val_dataset: Validation dataset
    """
    # Define transformations with improved augmentation
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        ], p=0.5),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
        ], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Create datasets
    train_dataset = PascalVOCDataset(
        root=root,
        year="2012",
        split="train",
        transform=train_transform,
        download=False,
        keep_difficult=False
    )

    val_dataset = PascalVOCDataset(
        root=root,
        year="2012",
        split="val",
        transform=val_transform,
        download=False,
        keep_difficult=True
    )

    # Define custom collate function
    def custom_collate_fn(batch):
        """Custom collate function that handles variable-sized metadata."""
        # Filter out any bad samples (return None for any tensor)
        batch = [b for b in batch if b[0] is not None and b[1] is not None]

        # If no valid samples remain, return empty tensors
        if len(batch) == 0:
            return torch.zeros((0, 3, img_size, img_size)), torch.zeros((0, len(VOC_CLASSES))), []

        # Extract images and targets
        images = [item[0] for item in batch]
        targets = [item[1] for item in batch]

        # Keep metadata as a list without collating
        metadata = [item[2] for item in batch]

        # Stack images and targets (normal collation)
        images = torch.stack(images, 0)
        targets = torch.stack(targets, 0)

        return images, targets, metadata

    # Create samplers for distributed training
    train_sampler = DistributedSampler(train_dataset) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None

    # Create dataloaders
    loader_args = {
        'pin_memory': pin_memory,
        'collate_fn': custom_collate_fn
    }

    # Only add persistent_workers and prefetch_factor if num_workers > 0
    if num_workers > 0:
        loader_args['persistent_workers'] = persistent_workers
        loader_args['prefetch_factor'] = prefetch_factor

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        drop_last=drop_last,
        sampler=train_sampler,
        **loader_args
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        sampler=val_sampler,
        **loader_args
    )

    return train_loader, val_loader, train_dataset, val_dataset


def get_num_classes() -> int:
    """Return the number of classes in PASCAL VOC."""
    return len(VOC_CLASSES)


def get_class_names() -> List[str]:
    """Return the list of class names in PASCAL VOC."""
    return VOC_CLASSES


def get_class_counts() -> List[int]:
    """
    Return the list of sample counts per class.

    Returns:
        List of counts, or None if not computed yet.
    """
    global _class_counts
    return _class_counts

