# DAC-204-PROJECT-
Breast Ultrasound Image Segmentation
This project implements a U-Net model with a ResNeXt101-32x16d encoder for segmenting breast ultrasound images to identify tumor regions. The model is trained on the Breast Ultrasound Images Dataset (BUSI) to delineate benign, malignant, and normal regions, aiding medical diagnostics.
Features

Model: U-Net with ResNeXt101-32x16d encoder, pretrained on Instagram data.
Dataset: BUSI dataset with 780 image-mask pairs.
Loss Function: Dice loss for handling class imbalance.
Performance: Achieves F1 score of 0.80 and mAP of 0.79 on the test set.
Environment: Python 3.11, PyTorch, segmentation_models_pytorch, CUDA support.

Installation

Clone the repository:
git clone [link](https://github.com/laluprasad7/DAC-204-PROJECT-/blob/main/README.md)
cd breast-ultrasound-segmentation


Install dependencies:
pip install -r requirements.txt

Requirements include torch, torchvision, segmentation-models-pytorch, numpy, pandas, matplotlib, pillow, and tqdm.

Download the BUSI dataset and place it in ./data/Dataset_BUSI_with_GT/:

Breast Ultrasound Images Dataset



Usage

Prepare Data:

Ensure the dataset is structured with subfolders benign, malignant, and normal containing images and corresponding _mask.png files.


Train the Model:Run the Jupyter notebook resnext101-32x16d.ipynb:
jupyter notebook resnext101-32x16d.ipynb


The notebook preprocesses data, trains the model for 20 epochs, and saves the best model to best_model.pth.
Training uses a batch size of 8, Adam optimizer (lr=1e-4), and a combined loss (0.2 training + 0.8 testing).


Evaluate:

The notebook includes code to compute training and testing losses. Add evaluation code (e.g., using scikit-learn) to calculate precision, recall, F1 score, and mAP on the test set.


Visualize:

Extend the notebook to generate loss curves and segmentation outputs using matplotlib. Example outputs include training/testing loss plots and input/ground truth/predicted mask comparisons.



Project Structure

resnext101-32x16d.ipynb: Main notebook with data preprocessing, model training, and evaluation.
data/Dataset_BUSI_with_GT/: Directory for the BUSI dataset (not included; must be downloaded).
best_model.pth: Saved weights of the best model (generated after training).
requirements.txt: List of Python dependencies.

Results

Performance Metrics:
Precision: 0.80
Recall: 0.73
F1 Score: 0.77


Visualizations:
Loss curves show convergence with the best model at epoch 14 (combined loss: 0.2364).
Segmentation outputs demonstrate accurate tumor boundary detection with minor errors in complex cases.



Future Improvements

Add data augmentation (e.g., rotation, flipping) to enhance robustness.
Implement ensemble models for improved boundary precision.
Include additional evaluation metrics and visualization scripts.

References

BUSI Dataset
Segmentation Models PyTorch
U-Net Paper
ResNeXt Paper

