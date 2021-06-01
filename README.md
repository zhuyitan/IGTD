# Image Generator for Tabular Data (IGTD): Converting Tabular Data into Images for Deep Learning Using Convolutional Neural Networks

## Description

Image Generator for Tabular Data (IGTD) is an algorithm for transforming tabular data into images. The algorithm assigns each feature to a unique pixel position in the image representation. Similar features are assigned to neighboring pixels, while dissimilar features are assigned to pixels that are far apart. According to the assignment, an image is generated for each sample, in which the pixel intensity reflects the value of the corresponding feature in the sample. One of the most important applications for the generated images is to build Convolutional Neural Networks (CNNs) based on the image representations in subsequent analysis. A publication about the IGTD algorithm is available at https://www.nature.com/articles/s41598-021-90923-y

## User Community

- Primary: machine learning; computational data modeling
- Secondary: bioinformatics; computational biology

## Usability

To use this software package, users must possess the basic skills to program and run Python scripts. Users need to process the input data into the data format accepted by the package. Users also need to understand the input parameters of the IGTD algorithm, so that the parameters can be appropriately set to execute the algorithm.

## Uniqueness

IGTD is a novel algorithm for transforming tabular data into images. Compared with existing methods for converting tabular data into images, IGTD has several advantages. 
- IGTD does not require prior knowledge about the features. Thus, it can be used in the absence of domain knowledge. 
- IGTD generates compact image representations, in which each pixel represents a unique feature. Deep learning based on compact image representations usually requires less memory and time to train the prediction model.
- IGTD has been shown to generate compact image representations promptly, which also better preserve the feature neighborhood structure.
- CNNs trained on IGTD images achieve a better (or similar) prediction performance than both CNNs trained on alternative image representations and prediction models trained on the original tabular data. 
- IGTD provides a flexible framework that can be extended to accommodate diversified data and requirements. The size and shape of the image representation can be flexibly chosen.  

## Components

The package includes two Python scripts. 
1. IGTD_Functions.py provides all the functions used by the IGTD algorithm. It provides comments explaining the input and output of each function.
2. Example_Run.py provides examples showing how to run the IGTD algorithm for demo purpose.

The package also includes a small dataset for demonstrating its utility, which is a gene expression dataset including 100 cancer cell lines and 1600 genes. 

## Technical Details

Refer to this [README](https://github.com/zhuyitan/IGTD/tree/main/Scripts).
