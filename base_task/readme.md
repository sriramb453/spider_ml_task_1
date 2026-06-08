Base Task: Fashion-MNIST Image Classification

A custom Convolutional Neural Network (CNN) built from scratch using PyTorch to classify clothing items from the Fashion-MNIST dataset.

Model Architecture & Training Pipeline

Architecture: Features two primary Convolutional blocks (Conv2d ➔ BatchNorm ➔ ReLU ➔ MaxPool ➔ Dropout) followed by a fully connected linear classifier.

Optimizer:`AdamW` with weight decay (`1e-4`) for robust regularization.
Learning Rate Scheduling:`ReduceLROnPlateau` dynamically reduces the learning rate when validation loss stops improving.

Overfit Prevention: Implements Data Augmentation (Random Horizontal Flips, Random Cropping) alongside Early Stopping (Patience = 5).

Hardware:Configured to automatically detect and utilize CUDA if available, falling back to CPU otherwise.

Directory Contents
* `notebooks/Implementation code.ipynb`: The core Jupyter Notebook containing the full data loading, model training, evaluation, and inference pipeline.
* `saved_models/`: Contains the optimized model weights (`best_fashion_model.pth` and `fashion_model_weights.pkl`) saved from the highest-performing epoch.
* `accuracy_loss_plots.png`: Visual representation of the training and validation curves.
* `submission.csv`: The final model predictions mapping indexes to their predicted class labels for the 10,000-image test set.

How to Run

1. Install Requirements
Ensure you have PyTorch and standard data science libraries installed:
`pip3 install torch torchvision pandas matplotlib numpy`

2. Execute the Notebook
Navigate to the `base_task/notebooks/` directory and open `Implementation code.ipynb` using Jupyter Notebook, JupyterLab, or Visual Studio Code.

3. Run the Pipeline
Execute all cells in sequence. The script will automatically download the dataset (if missing), train the CNN, save the best weights, generate the performance plots, and output a fresh `submission.csv`.