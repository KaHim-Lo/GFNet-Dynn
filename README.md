# Lightweight Remote Sensing Scene Classification on Edge Devices via Knowledge Distillation and Early-exit

**Read this in other languages: [English](README.md), [中文](README_zh.md).**

GFNet-Dynn is a hybrid deep learning model that combines the advantages of GFNet and Dynn architectures. This repository contains the codebase for training, evaluating, and deploying GFNet-Dynn models. The project aims to improve the efficiency and accuracy of deep neural networks by leveraging a dynamic early-exit mechanism.

## Installation Guide

### 1. **Clone the Repository**

```bash
git clone https://github.com/yourusername/GFNet-Dynn.git
cd GFNet-Dynn
```

### 2. **Install Dependencies**

Refer to the dependency installation guides from [GFNet](https://github.com/raoyongming/GFNet) and [Dynn](https://github.com/networkslab/dynn).

### 3. **Load Data**

Place the datasets in the `data/` folder at the same level as this project. For specific data loading instructions, refer to the `load_dataset` function in `main_dynn.py`.
Supported datasets include but are not limited to:
- AID
- NaSC
- NWPU
- PatternNet
- UCM

Ensure that the dataset paths are correctly configured so the code can load the data properly. (You can check dataset configurations in `data_loader_help.py`.)

### 4. **Configure Checkpoints**

Use the `--resume` argument to choose whether to load a model checkpoint. You can configure the `checkpoint_path` in the `main` function of `main_dynn.py` to resume training or perform model evaluation.

## Usage

### Model Training

To train the GFNet-Dynn model, you can use the provided Bash script:

```bash
./main_dynn.bash
```

Alternatively, refer to the command examples in `main_dynn.txt` to learn how to configure different parameters for training.

### Model Evaluation

For evaluating pre-trained models, configure the `--eval` and `--resume` parameters, and specify the model checkpoint path `checkpoint_path` to run the evaluation function.

### Experiment Tracking

This project uses MLflow for experiment tracking. Start the MLflow server to track experiments:

```bash
mlflow ui
```

Then, access `http://localhost:5000` in your browser and navigate to the `mlruns/` directory to view and compare different runs.

## Configuration Details

The main configuration parameters of GFNet-Dynn can be found in `main_dynn.py`. Below are some key parameters:

- `--data_set`: Specifies the dataset name (e.g., AID, NaSC, NWPU).
- `--img_size`: Sets the input image size.
- `--num_classes`: Specifies the number of classes for the classification task.
- `--ce_ic_tradeoff`: Controls the trade-off coefficient between classification loss and intermediate classifier loss.
- `--max_warmup_epoch`: Sets the maximum number of training epochs for the dynamic warmup phase.
- `--resume`: Loads a pre-trained model or resumes training.
- `--eval`: Enables evaluation mode for validation set testing only.

For more detailed configuration information, refer to the `argparse` configuration section in `main_dynn.py`.

## Model Training Workflow

The training workflow of GFNet-Dynn is mainly divided into two phases:

1. **Dynamic Warmup Phase**:
   - In this phase, the model gradually trains each intermediate classifier (exit) and freezes the best-performing classifier based on validation accuracy.
   - Once an exit classifier fails to improve on the validation set for `patience` consecutive epochs, it will be frozen, and its optimal weights will be saved.
   - If all exit classifiers converge, training will be terminated early.

2. **Full Model Training Phase**:
   - After the warmup phase, the model enters the full model training phase.
   - The weights of all exit classifiers will be fixed, and only the backbone network will be trained to further improve overall performance.
   - The final model will include multiple exit classifiers, allowing dynamic early-exit decisions during inference based on specific needs.

## Checkpoint File Description

GFNet-Dynn saves model checkpoint files during training for resuming training or performing model evaluation.

All checkpoint files are saved by default in the `checkpoint/` directory, which contains multiple subfolders for organizing model files from different phases and configurations:

- **`best/`**  
  Stores the best-performing model files on the validation set. This is the final recommended model weight after training (added **manually**).

- **`_warmup/`**  
  Stores model checkpoints from the dynamic warmup phase, used for training and freezing intermediate classifiers (exit).

- **`checkpoint_{dataset}_{ce_ic_tradeoff}_confEE/`**  
  Stores model files trained under different dataset (`{dataset}`) and loss trade-off coefficient (`{ce_ic_tradeoff}`) configurations, allowing for differentiation between experimental conditions.

For more details on the checkpoint file saving paths and naming conventions, refer to the relevant implementation in `our_train_helper.py`.

## References

This project draws inspiration from the following repositories:

- [GFNet](https://github.com/raoyongming/GFNet)
- [Dynn](https://github.com/networkslab/dynn)

Please consult these repositories for more detailed information on the underlying architecture and methods.
