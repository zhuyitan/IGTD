## Method Description

Image Generator for Tabular Data (IGTD) transforms tabular data into images. The algorithm assigns each feature to a pixel in the image. According to the assignment, an image is generated for each data sample, in which the pixel intensity reflects the value of the corresponding feature in the sample. The algorithm searches for an optimized assignment of features to pixels by minimizing the difference between the ranking of pairwise distances between features and the ranking of pairwise distances between the assigned pixels, where the distances between pixels are calculated based on their coordinates in the image. Minimizing the difference between the two rankings assigns similar features to neighboring pixels and dissimilar features to pixels that are far apart. The optimization is achieved through an iterative process of swapping the pixel assignments of two features. In each iteration, the algorithm identifies the feature that has not been considered for swapping for the longest time, and seeks for a feature swapping for it that reduces the difference between the two rankings most.   

## Setup

To set up the Python environment needed to run this algorithm:
1. Install [conda](https://docs.conda.io/en/latest/) package manager.
2. Clone this repository.
3. Enter the directory of Scripts
4. Create the environment as shown below.
    ```
    conda env create -f environment.yml -n IGTD
    conda activate IGTD
    ```
5.  Run the Example_Run.py script for demo.

## Use IGTD to Convert Tabular Data into Images

The IGTD_Functions.py script provides all the functions used by the IGTD algorithm. Please see comments in the script for explanations about the inputs and outputs of all functions. table_to_image is the main function for running the IGTD algorithm. min_max_transform is the function to preprocess input data, so that the minimum and maximum values of each feature are linearly scaled to 0 and 1, respectively. Example_Run.py provides exmaples demonstrating how to use the functions for running the IGTD algorithm. 