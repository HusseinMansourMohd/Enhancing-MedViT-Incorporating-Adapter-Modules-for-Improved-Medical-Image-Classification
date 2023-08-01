# Enhancing MedViT: Incorporating Adapter Modules for Improved Medical Image Datasets.

### Description:

This project aims to improve the performance of the MedViT model, a robust Vision Transformer (ViT) optimized for small medical datasets, by incorporating Adapter Modules. Unlike the typical application of ViT which requires large datasets, our implementation focuses on maximizing performance with smaller, more specific datasets that are typical in medical applications.

Adapters are small neural network modules that are introduced in between the main layers of a transformer-based model. They enable the model to adapt its internal representations to better cater to the task or dataset. By adding these adapters, we aim to increase the model's flexibility and capacity to learn complex features, thereby enhancing its classification performance.

This project involves modifying the original MedViT model architecture to include these adapter layers, training the adapted model with our specific medical datasets, and assessing the impact of these additions on model performance in terms of classification accuracy, precision, and recall.

Through this project, we aim to demonstrate the effectiveness of adapter modules in improving the performance of Vision Transformers, even when dealing with smaller datasets. This approach will provide significant benefits in the field of medical image analysis, enabling more accurate and reliable diagnoses and predictions.

### MedNIST Dataset:
The MedNIST dataset is a collection of medical images spanning six categories. These categories include Hand, Abdomen, Chest, HeadCT, BreastMRI, and CXR (Chest X-ray). It is a balanced dataset, with an equal number of images in each category, which makes it a suitable choice for classification tasks.

The MedNIST dataset was compiled from several larger, publicly available datasets, including the RSNA Pneumonia Detection Challenge dataset and the NIH Chest X-ray dataset, among others.

Here are some more details about the dataset:

Number of Classes: The MedNIST dataset includes six classes: Hand, Abdomen, Chest, HeadCT, BreastMRI, and CXR (Chest X-ray).

Number of Images: There are typically 10,000 images for each class, resulting in a total of approximately 60,000 images.

Image Size: All the images are 64x64 grayscale images.

Balance of Classes: The dataset is balanced, meaning it contains an equal number of images for each class.

Preprocessing: The images have been preprocessed to a standard size and grayscale format for easier use.

The MedNIST dataset is commonly used in medical image classification tasks as it covers a range of imaging types found in the medical field. It is also an easily accessible dataset that is helpful for training and validating machine learning models, especially for those new to medical image analysis. However, it should be noted that real-world medical datasets are often much more complex and unbalanced.


