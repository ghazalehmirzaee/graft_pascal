"""
PASCAL VOC dataset implementation for GRAFT framework.
"""
import os
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple, Optional, Callable

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# PASCAL VOC class names (20 classes)
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]


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
            keep_difficult: bool = False
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
        """
        self.root = root
        self.year = year
        self.split = split
        self.transform = transform
        self.keep_difficult = keep_difficult

        self.image_dir = os.path.join(root, f"VOC{year}", "JPEGImages")
        self.anno_dir = os.path.join(root,  f"VOC{year}", "Annotations")
        self.split_dir = os.path.join(root, f"VOC{year}", "ImageSets", "Main")

        # If download is True and the dataset doesn't exist, download it
        if download and not os.path.exists(self.image_dir):
            self._download()

        # Read image IDs from the specified split
        self.ids = self._read_image_ids()

        # Compute class weights for handling class imbalance
        self.class_weights = self._compute_class_weights()

        # Build co-occurrence statistics
        self.co_occurrence = self._build_co_occurrence_matrix()

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
        img = Image.open(img_path).convert("RGB")

        # Load annotations
        anno_path = os.path.join(self.anno_dir, f"{img_id}.xml")
        boxes, labels, difficulties = self._parse_annotation(anno_path)

        # Create multi-hot encoded label tensor
        target = torch.zeros(len(VOC_CLASSES))
        for label in labels:
            if difficulties[labels.index(label)] and not self.keep_difficult:
                continue
            target[label] = 1

        # Prepare metadata with bounding boxes and image info
        metadata = {
            'image_id': img_id,
            'boxes': boxes,
            'labels': labels,
            'difficulties': difficulties,
            'height': img.height,
            'width': img.width
        }

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
            tree = ET.parse(anno_path)
            root = tree.getroot()

            for obj in root.findall("object"):
                label_name = obj.find("name").text
                if label_name in VOC_CLASSES:
                    class_counts[VOC_CLASSES.index(label_name)] += 1

        # Compute weights as inverse of frequency
        class_weights = 1.0 / torch.max(class_counts, torch.ones_like(class_counts))

        # Normalize weights
        class_weights = class_weights / torch.sum(class_weights) * len(VOC_CLASSES)

        return class_weights

    def _build_co_occurrence_matrix(self) -> torch.Tensor:
        """
        Build co-occurrence matrix for all classes.

        Returns:
            Co-occurrence matrix of shape (num_classes, num_classes)
        """
        num_classes = len(VOC_CLASSES)
        co_occurrence = torch.zeros((num_classes, num_classes))

        # Count co-occurrences of classes
        for img_id in self.ids:
            anno_path = os.path.join(self.anno_dir, f"{img_id}.xml")
            tree = ET.parse(anno_path)
            root = tree.getroot()

            # Collect all labels in this image
            labels = []
            for obj in root.findall("object"):
                label_name = obj.find("name").text
                if label_name in VOC_CLASSES:
                    labels.append(VOC_CLASSES.index(label_name))

            # Update co-occurrence matrix
            for i in labels:
                for j in labels:
                    co_occurrence[i, j] += 1

        return co_occurrence

    def get_spatial_statistics(self) -> Dict[str, torch.Tensor]:
        """
        Calculate spatial statistics for all classes.

        Returns:
            Dictionary with spatial statistics:
                - positions: Average center positions for each class
                - sizes: Average sizes for each class
                - overlaps: Average overlap between classes
        """
        num_classes = len(VOC_CLASSES)

        # Initialize statistics
        positions = torch.zeros((num_classes, 2))  # x, y center positions
        sizes = torch.zeros((num_classes, 2))  # width, height
        counts = torch.zeros(num_classes)
        overlaps = torch.zeros((num_classes, num_classes))

        # Process all images
        for img_id in self.ids:
            anno_path = os.path.join(self.anno_dir, f"{img_id}.xml")
            tree = ET.parse(anno_path)
            root = tree.getroot()

            width = float(root.find("size").find("width").text)
            height = float(root.find("size").find("height").text)

            # Collect all objects in this image
            objects = []
            for obj in root.findall("object"):
                label_name = obj.find("name").text
                if label_name not in VOC_CLASSES:
                    continue

                bbox = obj.find("bndbox")
                x_min = float(bbox.find("xmin").text) / width
                y_min = float(bbox.find("ymin").text) / height
                x_max = float(bbox.find("xmax").text) / width
                y_max = float(bbox.find("ymax").text) / height

                label_idx = VOC_CLASSES.index(label_name)

                # Update position and size statistics
                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2
                obj_width = x_max - x_min
                obj_height = y_max - y_min

                positions[label_idx, 0] += center_x
                positions[label_idx, 1] += center_y
                sizes[label_idx, 0] += obj_width
                sizes[label_idx, 1] += obj_height
                counts[label_idx] += 1

                objects.append({
                    'label': label_idx,
                    'bbox': [x_min, y_min, x_max, y_max]
                })

            # Calculate overlaps between objects
            for i, obj1 in enumerate(objects):
                for j, obj2 in enumerate(objects):
                    if i != j:
                        bbox1 = obj1['bbox']
                        bbox2 = obj2['bbox']
                        label1 = obj1['label']
                        label2 = obj2['label']

                        # Calculate IoU
                        x_min_overlap = max(bbox1[0], bbox2[0])
                        y_min_overlap = max(bbox1[1], bbox2[1])
                        x_max_overlap = min(bbox1[2], bbox2[2])
                        y_max_overlap = min(bbox1[3], bbox2[3])

                        if x_max_overlap > x_min_overlap and y_max_overlap > y_min_overlap:
                            overlap_area = (x_max_overlap - x_min_overlap) * (y_max_overlap - y_min_overlap)
                            bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
                            bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
                            iou = overlap_area / (bbox1_area + bbox2_area - overlap_area)

                            overlaps[label1, label2] += iou

        # Normalize statistics
        for i in range(num_classes):
            if counts[i] > 0:
                positions[i] /= counts[i]
                sizes[i] /= counts[i]

        for i in range(num_classes):
            for j in range(num_classes):
                if i != j and self.co_occurrence[i, j] > 0:
                    overlaps[i, j] /= self.co_occurrence[i, j]

        return {
            'positions': positions,
            'sizes': sizes,
            'overlaps': overlaps
        }

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
        std: List[float] = [0.229, 0.224, 0.225]
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

    Returns:
        Tuple containing:
            - train_loader: DataLoader for training set
            - val_loader: DataLoader for validation set
            - train_dataset: Training dataset
            - val_dataset: Validation dataset
    """
    # Define transformations
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
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
        # Extract images and targets
        images = [item[0] for item in batch]
        targets = [item[1] for item in batch]

        # Keep metadata as a list without collating
        metadata = [item[2] for item in batch]

        # Stack images and targets (normal collation)
        images = torch.stack(images, 0)
        targets = torch.stack(targets, 0)

        return images, targets, metadata

    # Create dataloaders with custom collate function
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate_fn
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=custom_collate_fn
    )

    return train_loader, val_loader, train_dataset, val_dataset


def get_num_classes() -> int:
    """Return the number of classes in PASCAL VOC."""
    return len(VOC_CLASSES)


def get_class_names() -> List[str]:
    """Return the list of class names in PASCAL VOC."""
    return VOC_CLASSES

