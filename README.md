# PHMG-Net: A Plum Grading Method Based on Persistent Homology Gating and Multimodal Feature Fusion

This repository contains the official PyTorch implementation of the paper **PHMG-Net: A Plum Grading Method Based on Persistent Homology Gating and Multimodal Feature Fusion**. The network fuses RGB images, physical properties (weight, sugar content) and topological features extracted via persistent homology to achieve automatic quality grading of plums.

## Model Overview
PHMG-Net is built on the Swin Transformer V2 backbone, with core features including:
- A multimodal feature fusion module that integrates visual features, weight/sugar scalar features, and persistent homology topological features
- A persistent homology gating mechanism
- Low-rank adaptation (LoRA) parameter-efficient fine-tuning strategy

The model classifies plums into 4 categories:
| Class ID | Category | Filename prefix |
|----------|----------|-----------------|
| 0 | Large fruit | `d*` |
| 1 | Medium fruit | `z*` |
| 2 | Small fruit | `x*` |
| 3 | Defective fruit | `h*` |

## Requirements
- Python >= 3.8
- PyTorch >= 1.10
- torchvision
- pandas
- scikit-learn
- scikit-image
- numpy
- tqdm
- openpyxl
- gudhi (for persistent homology diagram generation)

Install dependencies:
```bash
pip install torch torchvision pandas scikit-learn scikit-image numpy tqdm openpyxl gudhi
```

## Dataset Structure
Organize your dataset in the `data/` directory as follows:
```
data/
├── data/                          # Raw RGB plum images
│   ├── d100.jpg
│   ├── z15.jpg
│   └── ...
├── persistent_diagram/            # Persistent homology diagrams (.npy format)
│   ├── d100/
│   │   └── pd_combined.npy
│   ├── z15/
│   │   └── pd_combined.npy
│   └── ...
├── plum_dataset_train.xlsx        # Training split annotation
├── plum_dataset_val.xlsx          # Validation split annotation
└── plum_dataset_test.xlsx         # Test split annotation
```

Five-fold cross-validation is performed on the full dataset to split training, validation and test sets for robust model performance evaluation.

Each Excel annotation file contains 3 columns:
- `path`: image filename without extension (e.g. `d100`)
- `weight`: measured fruit weight (unit: g)
- `sugar`: measured soluble solids content (SSC, unit: %) of the fruit

## Training
Run the training script with default parameters:
```bash
python train.py
```

### Training Arguments
| Argument | Type | Default Value | Description |
|----------|------|---------------|-------------|
| `--input_size` | int | 128 | Input image resolution |
| `--classes_num` | int | 4 | Number of classification categories |
| `--lr` | float | 0.005 | Initial learning rate |
| `--alpha` | float | 0.1 | Weight of the topological auxiliary loss term |
| `--class_counts` | str | `474,1152,954,492` | Number of samples per class |
| `--modes` | list | `weight sugar persistent` | Input modalities used for training |
| `--early_stop_patience` | int | 15 | Early stopping patience (consecutive epochs without improvement) |

Example custom training command:
```bash
python train.py --batch_size 128 --lr 0.005  --modes weight sugar persistent
```

## Output
- The best trained model will be automatically saved to `checkpoints/best_model_wsp.pth`
- The training log prints accuracy, AUC, F1-score and mAP metrics for both validation and test sets per epoch

## Notes
Only a representative subset of the dataset is provided in this repository. The complete dataset can be requested after the paper is published.
