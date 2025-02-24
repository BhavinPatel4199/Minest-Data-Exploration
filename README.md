# Minest-Data-Exploration

## Overview
This project focuses on classifying handwritten digits from the MNIST dataset using various machine learning models. The dataset consists of grayscale images of size 28x28, representing digits (0-9). The objective is to train a model that accurately predicts the correct digit based on the input image.

## Dataset
The MNIST dataset contains:
- **60,000 training images**
- **10,000 testing images**
Each image is a 28x28 pixel grayscale representation of a digit, and the dataset is provided in binary format.

## Project Structure
```
├── data/                         # Contains the MNIST dataset files 4 files (train-images-idx3-ubyte.gz: training set images (9912422 bytes), train-labels-idx1-ubyte.gz: training set labels (28881 bytes), t10k-images-idx3-ubyte.gz: test set images (1648877 bytes), t10k-labels-idx1-ubyte.gz: test set labels (4542 bytes))
├── notebooks/                    # Jupyter Notebooks for exploration and model development
├── results/                      # Model outputs, evaluation metrics, and visualizations
├── src/                          # Python scripts for loading data, training models, and evaluation
├── README.md                     # Project documentation
└── requirements.txt              # List of dependencies
```

## Installation
To run this project, install the required dependencies:
```sh
pip install -r requirements.txt
```

## Steps in the Project

### 1. Data Preparation
- Loaded training and test images from binary files.
- Converted image data into numpy arrays.
- Normalized pixel values to the range [0, 1].
- Stored image dimensions in CSV files for reference.

### 2. Exploratory Data Analysis (EDA)
- Checked dataset shape, missing values, and duplicates.
- Visualized label distributions using histograms.
- Displayed sample images from training and test datasets.

### 3. Data Preprocessing
- Split the dataset into training (80%) and validation (20%) subsets.
- Applied Principal Component Analysis (PCA) to reduce dimensionality while retaining 95% variance.

### 4. Model Training
Implemented and trained the following models:
- **Support Vector Machine (SVM)** (RBF kernel, C=10, gamma=0.01)
- **Random Forest Classifier** (100 estimators)
- **K-Nearest Neighbors (KNN)** (k=3)

### 5. Model Evaluation
- Computed F1-score for model comparison.
- Selected the best-performing model (SVM with F1-score of 0.9846 on validation set).
- Evaluated the final model on the test dataset.
- Generated a classification report and confusion matrix.

## Results
- The best-performing model was **SVM**, achieving **98% accuracy** on the test set.
- The confusion matrix showed minimal misclassifications.

## Visualizations
- Histograms of class distributions.
- Sample images from the dataset.
- Confusion matrix of final model predictions.

## How to Run
1. Ensure dependencies are installed using `pip install -r requirements.txt`.
2. Run the Jupyter notebooks in `notebooks/` for step-by-step execution.
3. Execute `src/train_model.py` to train and evaluate models.
4. Check `results/` for output files and evaluation metrics.

## Future Improvements
- Fine-tune hyperparameters for better performance.
- Experiment with deep learning models (e.g., CNNs).
- Apply data augmentation techniques to enhance model generalization.

## Acknowledgments
- MNIST dataset provided by Yann LeCun.
- Libraries used: NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn.
