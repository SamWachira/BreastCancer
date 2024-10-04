# Breast Cancer Classification Using CNN

## Overview

This project demonstrates how to build and train a Convolutional Neural Network (CNN) to classify breast cancer images into three categories: benign, malignant, and normal. The dataset is retrieved from AWS S3, and the model is trained using TensorFlow and Keras. The process includes data preprocessing, handling class imbalance, building a CNN model, training with class weights, and visualizing model performance.


# Steps in the Notebook

## Step 1: Import Necessary Libraries

Libraries such as TensorFlow, Keras, and AWS SDK (boto3) are imported for data handling, model building, and downloading images from S3. Other essential libraries include NumPy for data manipulation and Matplotlib for visualization.

1. TensorFlow/Keras: For deep learning and building the CNN model.
2. Boto3: For downloading images from an S3 bucket.
3. NumPy: For numerical operations.
4. Matplotlib: For plotting training performance graphs.


## Step 2: Download Images from S3

This step downloads image datasets from an S3 bucket into a local directory. The script maintains the folder structure of the dataset, which helps in automatically labeling the images for training and validation.

- S3 Bucket: Specify the bucket name and path where images are stored.
- Local Directory: Define the local path where images will be saved.

## Step 3: Load and Split the Dataset

The images are loaded into a TensorFlow dataset using image_dataset_from_directory. The dataset is automatically labeled based on the folder structure (benign, malignant, normal) and split into training (80%) and validation (20%).

Image Size: 500x500 pixels.
Batch Size: 32.

## Step 4: Data Preprocessing

This step involves:
1. **Data Augmentation:** Randomly flip, rotate, zoom, and adjust brightness of images to prevent overfitting and improve model generalization.
2. **Normalization:** Pixel values are rescaled from the range [0, 255] to [0, 1] for faster training convergence.

## Step 5: Handle Class Imbalance

Due to the imbalance in the dataset (uneven number of images for each class), class weights are calculated and applied during training. This helps in giving more importance to the under-represented classes (malignant and normal).

- **Class Weights:** Computed based on the number of images in each class to avoid bias toward over-represented classes.

## Step 6: Build the CNN Model

The CNN model is built using Keras' Sequential API with multiple convolutional layers followed by max-pooling. The fully connected layers are added at the end, including a dropout layer for regularization.

- **Convolutional Layers:** Extract features from images.
- **Dense Layers:** Perform classification.
- **Dropout Layer:** Prevent overfitting by randomly setting units to 0 during training.

# Step 7: Train the Model

The model is trained using the Adam optimizer and categorical crossentropy loss function. Class weights are used to address class imbalance. The model is trained for 20 epochs, though this number can be adjusted based on performance.

## Step 8: Evaluate the Model

After training, the model is evaluated on the validation dataset to determine its performance, particularly the accuracy and loss.

# Step 9: Save the Model
The trained model is saved in .h5 format for later use.

# Conclusion
This notebook provides a comprehensive guide for building and training a CNN for breast cancer classification. The workflow includes downloading images from S3, data preprocessing, handling class imbalance, model training, evaluation, and visualization of the results.

**Note:** The final accuracy of the model is around 54%, which can be improved by further optimization, model tuning, or using pre-trained models for transfer learning.

# Prerequisites
- AWS S3 access with stored image datasets.
- TensorFlow, Keras, and other Python libraries installed in your environment.
- SageMaker or a similar environment for cloud-based training.
# Instructions
1. Clone this notebook and place your breast cancer images in the specified S3 bucket.
2. Adjust the bucket_name and prefix to point to your dataset.
3. Run each cell step-by-step to preprocess the data, build, train, and save the model.
4. (Optional) Fine-tune the model by adjusting parameters such as epochs, batch size, or adding more complex architectures.

This README provides a high-level overview and step-by-step guidance for running the breast cancer classification project.