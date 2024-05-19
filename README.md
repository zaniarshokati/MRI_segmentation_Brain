

# Brain Tumor Segmentation using U-Net with EfficientNetB7

This repository contains a PyTorch implementation of a U-Net model for brain tumor segmentation, leveraging an EfficientNetB7 encoder. The project is structured to facilitate the training, evaluation, and visualization of the segmentation performance on MRI images, assisting in the identification and analysis of brain tumors.

## Project Structure

- **`main.py`**: The primary script which encapsulates the model definition, training procedures, evaluation metrics, and visualization tools.
- **`utils.py`**: Includes utility classes and functions such as `MriDataset` for dataset handling and `EarlyStopping` for enhanced training performance.

## Features

- **Model**: Employs the U-Net architecture with an EfficientNetB7 encoder, pre-trained on the ImageNet dataset.
- **Metrics**: Utilizes Intersection over Union (IoU) and Dice coefficient to measure segmentation performance.
- **Data Augmentation**: Implements various augmentation techniques to enhance model robustness against overfitting.
- **Visualization**: Provides functions for visualizing training progress, dataset samples, and model predictions.

## Setup

### 1. Clone the Repository
```bash
git clone https://github.com/zaniarshokati/MRI_segmentation_Brain.git
cd brain-tumor-segmentation
```

## How to Contribute

Contributions are welcome! Feel free to fork this repository, make your improvements or customize the model and scripts, and submit pull requests with your enhancements.

## Credits

This implementation is inspired by and adapted from [Abdallah Wagih's work on Kaggle](https://www.kaggle.com/code/abdallahwagih/brain-tumor-segmentation-unet-efficientnetb7/comments), which provides a detailed approach to using the U-Net with the EfficientNetB7 encoder for segmenting brain tumors.

## License

This project is made available under the MIT License.

