# GRAFT: Graph-Augmented Framework with Vision Transformers for Multi-Label Classification

This repository implements the GRAFT framework for multi-label image classification, focusing on demonstrating the framework's generalizability across domains by adapting it to the PASCAL VOC dataset.

## Overview

GRAFT (Graph-Augmented Framework with Vision Transformers) combines the power of Vision Transformers (ViT) for feature extraction with graph-based relational learning to capture complex relationships between labels. The framework consists of two main stages:

1. **Self-supervised pre-training**: Leveraging Vision Transformers pre-trained with Masked Autoencoder (MAE) to capture robust and transferable image features without requiring extensive labeled data.

2. **Graph-based relationship modeling**: Integrating multiple specialized graphs to capture different types of inter-label dependencies, including:
   - Co-occurrence patterns
   - Spatial relationships
   - Visual feature similarities
   - Semantic relationships

## Project Structure

```
graft-pascal/
│
├── config/                       # Configuration files for different phases
│   ├── default.yaml              # Base configuration
│   └── phases/                   # Phase-specific configs
│
├── data/                         # Dataset handling
│   ├── pascal_voc.py             # PASCAL VOC dataset class
│   └── transforms.py             # Data transformations
│
├── models/                       # Model architecture
│   ├── vit.py                    # Vision Transformer backbone
│   ├── graph/                    # Graph components
│   │   ├── co_occurrence.py      # Co-occurrence graph
│   │   ├── spatial.py            # Spatial relationship graph
│   │   ├── visual.py             # Visual feature graph
│   │   ├── semantic.py           # Semantic relationship graph
│   │   └── fusion.py             # Graph fusion network
│   └── graft.py                  # Complete GRAFT model
│
├── utils/                        # Utility functions
│   ├── metrics.py                # Evaluation metrics
│   ├── visualization.py          # Visualization tools
│   ├── loss.py                   # Loss functions
│   └── logger.py                 # Logging utilities
│
├── train.py                      # Training script
├── eval.py                       # Evaluation script
└── README.md                     # Project documentation
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/graft-pascal.git
   cd graft-pascal
   ```

2. Create a conda environment and install dependencies:
   ```bash
   conda create -n graft python=3.10
   conda activate graft
   pip install torch torchvision torchaudio
   pip install matplotlib seaborn networkx scikit-learn pyyaml wandb
   ```

## Data Preparation

1. Download the PASCAL VOC 2012 dataset from [the official website](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html).

2. Extract the dataset to a directory of your choice.

3. Update the dataset path in the config file (`config/default.yaml`):
   ```yaml
   dataset:
     root: "/path/to/PASCAL"  # Update with your actual path
   ```

## Pre-trained Weights

1. Download the MAE pre-trained ViT-Base/16 weights from [the official repository](https://github.com/facebookresearch/mae) or use [the direct link](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth).

2. Place the pre-trained weights in the main project directory.

3. Update the path in the config file (`config/default.yaml`):
   ```yaml
   model:
     backbone:
       pretrained_weights: "./mae_pretrained_vit_base.pth"  # Update if needed
   ```

## Training

The training process follows a progressive approach with five distinct phases:

1. **Phase 1: Backbone Initialization** - Load pre-trained ViT-Base/16 weights
2. **Phase 2: Fine-tune Backbone** - Create vision-only baseline for comparison
3. **Phase 3: Graph Construction** - Build all graph components without training
4. **Phase 4: Progressive Graph Integration** - Fine-tune with all graph components
5. **Phase 5: Model Refinement** - Final refinement of the complete model

To train the model through all phases:

```bash
python train.py --config config/default.yaml
```

To start from a specific phase:

```bash
python train.py --config config/default.yaml --start_phase phase2_finetune
```

## Training on GPU Server

To train on a GPU server with SLURM workload manager, create a submission script:

```bash
#!/bin/bash
#SBATCH --job-name=graft-pascal
#SBATCH --output=graft-pascal_%j.out
#SBATCH --error=graft-pascal_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=64G

module load cuda/12.2
module load anaconda3

source activate graft

cd /path/to/graft-pascal
python train.py --config config/default.yaml
```

Submit the job:

```bash
sbatch submit_job.sh
```

## Evaluation

To evaluate a trained model:

```bash
python eval.py --config config/default.yaml --checkpoint outputs/checkpoints/final_model.pth --output_dir outputs/evaluation
```

This will generate various visualizations and metrics in the specified output directory.

## Results

The GRAFT framework demonstrates significant improvements over vision-only baselines by capturing complex label relationships. Key metrics include:

- Higher mean Average Precision (mAP)
- Improved F1 score
- Better handling of class imbalance
- Enhanced performance on rare classes

## GitHub Integration

To push your code to GitHub:

1. Create a new repository on GitHub.

2. Initialize git in your local project directory:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: GRAFT implementation for PASCAL VOC"
   ```

3. Connect to your GitHub repository:
   ```bash
   git remote add origin https://github.com/yourusername/graft-pascal.git
   git push -u origin master
   ```

## Examples

### Training Curves
The training process generates training curves that show the progression of loss and mAP across epochs:

```python
# Example code to display training curves
from utils.logger import GRAFTLogger

logger = GRAFTLogger(output_dir="./outputs", project_name="GRAFT-PASCAL")
logger.plot_training_curves()
```

### Graph Visualizations
You can visualize the graph relationships between classes:

```python
# Example code to visualize graph relationships
from utils.visualization import plot_adjacency_matrix
import torch

# Load model and get adjacency matrix
adj_matrix = model.graph_components["co_occurrence"].get_adjacency_matrix()
plot_adjacency_matrix(adj_matrix, class_names, title="Co-occurrence Graph")
```

## Citation

[//]: # (If you find this code useful for your research, please cite the original GRAFT paper:)

[//]: # ()
[//]: # (```bibtex)

[//]: # (@inproceedings{mirzaee2025graft,)

[//]: # (  title={GRAFT: Graph-Augmented Framework with Vision Transformers for Multi-Label Classification},)

[//]: # (  author={Mirzaee, Ghazaleh and Le, Ngan and Doretto, Gianfranco and Adjeroh, Donald},)

[//]: # (  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition &#40;CVPR&#41;},)

[//]: # (  year={2025})

[//]: # (})

[//]: # (```)

## License

[//]: # (This project is licensed under the MIT License - see the LICENSE file for details.)

## Acknowledgments

[//]: # ()
[//]: # (- The PASCAL VOC dataset team for providing the benchmark dataset)

[//]: # (- The authors of the Vision Transformer and Masked Autoencoder for their foundational work)
[//]: # ()
[//]: # (# graft_pascal)

[//]: # (# graft_pascal)

[//]: # (# graft_pascal)

[//]: # (# graft_pascal)

[//]: # (# graft_pascal)

[//]: # (# graft_pascal)

[//]: # (# graft_pascal)

[//]: # (# graft_pascal)

[//]: # (# graft_pascal)

[//]: # (# graft_pascal)

[//]: # (# graft_pascal)
# graft_pascal
# pascal_multistage
