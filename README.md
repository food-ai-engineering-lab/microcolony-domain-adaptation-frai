# microcolony-domain-adaptation-frai

This repository is for the article "Enhancing AI microscopy for foodborne bacterial classification using adversarial domain adaptation to address optical and biological variability", published in Frontiers in Artificial Intelligence.


## Contents

- [Requirements](#requirements)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Grad-CAM Visualization](#grad-cam-visualization)
- [Example Outputs](#example-outputs)
- [Supplementary Scripts](#supplementary-scripts)
- [Acknowledgements](#acknowledgements)


## Requirements 

1. If you haven't done already, create a conda environment using the following command:
  
    ```
    $ conda create -n mcolony-classification python=3.8
    ```

2. Activate your conda environment named `mcolony-classification`: 

    ```
    $ conda activate mcolony-classification
    ```

3. Run the following command to install required packages:

    ```
    $ pip install -r requirements.txt
    ```


## Dataset

### Classes

- **Bc**: *Bacillus coagulans*
- **Bs**: *Bacillus subtilis*
- **Ec**: *Escherichia coli 1612*
- **Li**: *Listeria innocua*
- **SE**: *Salmonella enterica* Enteritidis
- **ST**: *Salmonella enterica* Typhimurium


### Data Split

The collected datasets were divided as follows:

- **Training Dataset**: 85% random split of the standard data.
- **Testing Dataset I**: 15% random split of the standard data.
- **Testing Dataset II**: A separate dataset collected under varying imaging conditions to evaluate the model's robustness.


### Imaging Conditions

Standard imaging conditions (source domain):

- Objective lens: 60x
- Microcolony ncubation time: 3 h
- Agar composition: Soft tryptic soy agar plates (0.7% w/v agarose)
- Imaging modality: Phase contrast microscopy
- Focus: Optimal

Variations (target domains):
- 20x-3h: Images captured using a 20x objective
- 20x-5h: Images captured using a 20x objective after 5 h of incubation
- brightfield: Images captured using brightfield microscopy


### Data Organization – Training Dataset

```bash
/mnt/data/mcolony-classification/train/ # root directory
├── Bc # class 1
│   ├── Bc-60x-3h-YYMMDD-001.jpg
│   ├── Bc-60x-3h-YYMMDD-002.jpg
│   ├── Bc-60x-3h-YYMMDD-003.jpg
│   └── ...
├── Bs # class 2
├── Ec # class 3
├── Li # class 4
├── SE # class 5
└── ST # class 6

```


### Data Organization – Testing Dataset I

```bash
/data/mcolony-classification-data/test1/ # root directory
├── Bc-60x-3h-YYMMDD-001.jpg # class 1
├── Bc-60x-3h-YYMMDD-002.jpg
├── Bc-60x-3h-YYMMDD-003.jpg
├── ...
├── Bs-60x-3h-YYMMDD-001.jpg # class 2
├── ...
└── ...

```


### Data Organization – Testing Dataset II

```bash
/data/mcolony-classification-data/test2/ # root directory
├── 20x-3h      # imaging condition 1
│   ├── Bc-20x-3h-YYMMDD-001.jpg # class 1
│   ├── Bc-20x-3h-YYMMDD-002.jpg
│   ├── Bc-20x-3h-YYMMDD-003.jpg
│   ├── ...
│   ├── Bs-20x-3h-YYMMDD-001.jpg # class 2
│   ├── ...
│   └── ...
├── 20x-5h      # imaging condition 2
├── brightfield # imaging condition 3

```


## Model Training

```bash
usage: train.py [-h] -r ROOT [-w WORKERS] [-b BATCH] [-g GPUS] [-a ACCUMULATION]

arguments:
  -h, --help            show help message
  -r, --root            Root folder of the dataset (required)
  -w, --workers         Number of dataloader workers per GPU (default: 5)
  -b, --batch           Batch size per GPU (default: 4)
  -g, --gpus            Number of GPUs (default: 1)
  -a, --accumulation    Number of accumulation steps (default: 0)
```

For example: `python train.py -r /path/to/train/dataset/`

**Learning curves:** PyTorch Lightning integrates seamlessly with TensorBoard, making it easy to track and visualize your training progress. After training, you can launch TensorBoard to visualize the logged metrics:

```bash
$ cd lightning_logs
$ tensorboard --logdir=.
```

Open the URL provided by TensorBoard in your web browser to view the `train_loss_epoch` and `val_loss_epoch` curves.


## Model Evaluation

```bash
usage: evaluate.py [-h] -r ROOT -c CKPT [-w WORKERS] [-b BATCH]

arguments:
  -h, --help            show help message
  -r, --root            Root folder of the dataset (required)
  -c, --ckpt            Path to checkpoint file (required)
  -w, --workers         Number of dataloader workers (default: 5)
  -b, --batch           Batch size (default: 1)
```

For example: `python evaluate.py -rt /path/to/train/dataset/ -r /path/to/test/dataset/ -c /path/to/checkpoint/ckpt_weights.ckpt`


## Grad-CAM Visualization

Grad-CAM (Gradient-weighted Class Activation Mapping) generates heatmaps to highlight important regions of an input image for model predictions.

- `visualize.ipynb`: Demonstrates how to use Grad-CAM to visualize model predictions.
- `gradcam_results.py`: Updated version by Aarham.

**NOTE (Pending): Data paths should be updated in these scripts**


## Example Outputs

`evaluate.py`:
- CSV file: Model predictions and ground truth labels.
- Confusion matrix: Visual representation of prediction accuracy for each class.


`gradcam_results.py`::
- Heatmap images: Highlight important regions of images for model predictions, showing model interpretation of microcolonies.


## Supplementary Scripts

- `architecture.py`: Defines the neural network architecture using EfficientNetV2 and PyTorch Lightning for microcolony classification.
- `datatools.py`: Provides data preprocessing, augmentation, and handling tools for microcolony datasets.
- `gradcam.py`: Implements Grad-CAM for visualizing model predictions.


## Acknowledgements

- USDA-NIFA [Grant 2021-67021-34256]
- USDA/NSF AI Institute for Next Generation Food Systems [Grant 2020-67021-32855]
- MSU Startup Funds
