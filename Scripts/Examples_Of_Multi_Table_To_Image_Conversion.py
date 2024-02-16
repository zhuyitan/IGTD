# This script provides examples to convert multiple data tables into multi-channel images, which can be modelled by
# multi-channel CNN models.

import pandas as pd
import os
from IGTD_Functions import min_max_transform, select_features_by_variation, multi_table_to_image
import numpy as np



num_row = 30        # Number of pixel rows in image representation
num_col = 30        # Number of pixel columns in image representation
num = num_row * num_col # Number of features to be included for analysis, which is also the total number of pixels in image representation
save_image_size = 3 # Size of pictures (in inches) saved during the execution of IGTD algorithm.

max_step = 30000    # The maximum number of iterations to run the IGTD algorithm, if it does not converge.
val_step = 300      # The number of iterations for determining algorithm convergence. If the error reduction rate
                    # is smaller than a pre-set threshold for val_step itertions, the algorithm converges.

# Import the example data and linearly scale each feature so that its minimum and maximum values are 0 and 1, respectively.
data1 = pd.read_csv('../Data/Example_Gene_Expression_Tabular_Data.txt', low_memory=False, sep='\t', engine='c',
                   na_values=['na', '-', ''], header=0, index_col=0)
# Select features with large variations across samples
id = select_features_by_variation(data1, variation_measure='var', num=num)
data1 = data1.iloc[:, id]

# Generate noise data and add it to the existing dataset to generate the second dataset.
add_noise = np.random.normal(loc=0.0, scale=0.2, size=data1.shape)
add_noise = pd.DataFrame(add_noise, index=data1.index, columns=data1.columns)
data2 = data1 + add_noise

# Generate noise data and add it to the existing dataset to generate the third dataset.
add_noise = np.random.normal(loc=0.0, scale=0.2, size=data1.shape)
add_noise = pd.DataFrame(add_noise, index=data1.index, columns=data1.columns)
data3 = data1 + add_noise

# Perform min-max transformation so that the maximum and minimum values of every feature become 1 and 0, respectively.
norm_data1 = min_max_transform(data1.values)
norm_data1 = pd.DataFrame(norm_data1, columns=data1.columns, index=data1.index)
norm_data2 = min_max_transform(data2.values)
norm_data2 = pd.DataFrame(norm_data2, columns=data2.columns, index=data2.index)
norm_data3 = min_max_transform(data3.values)
norm_data3 = pd.DataFrame(norm_data3, columns=data3.columns, index=data3.index)

image_dist_method = 'Euclidean'
error = 'abs'
result_dir = '../Results/Table_To_Image_Conversion/Test_1'
os.makedirs(name=result_dir, exist_ok=True)
# Run the IGTD algorithm using (1) three data tables together, (2) the Euclidean distance for calculating pairwise
# feature distances and pariwise pixel distances and (3) the absolute function for evaluating the difference
# between the feature distance ranking matrix and the pixel distance ranking matrix. Save the result in Test_1 folder.
multi_table_to_image(norm_d_list=(norm_data1, norm_data2, norm_data3), weight_list=[0.33, 0.33, 0.33],
                     fea_dist_method_list=('Euclidean', 'Euclidean', 'Euclidean'), scale=[num_row, num_col],
                     image_dist_method=image_dist_method, save_image_size=save_image_size,
                     max_step=max_step, val_step=val_step, normDir=result_dir, error=error,
                     switch_t=0, min_gain=0.000001)

# Run the IGTD algorithm using (1) two data tables, (2) the Pearson correlation coefficient for calculating
# pairwise feature distances, (3) the Manhattan distance for calculating pariwise pixel distances, and
# (4) the square function for evaluating the difference between the feature distance ranking matrix and
# the pixel distance ranking matrix. Save the result in Test_2 folder.
image_dist_method = 'Manhattan'
error = 'squared'
result_dir = '../Results/Table_To_Image_Conversion/Test_2'
os.makedirs(name=result_dir, exist_ok=True)
multi_table_to_image(norm_d_list=(norm_data1, norm_data2), weight_list=[0.5, 0.5],
                     fea_dist_method_list=('Pearson', 'Pearson'), scale=[num_row, num_col],
                     image_dist_method=image_dist_method, save_image_size=save_image_size,
                     max_step=max_step, val_step=val_step, normDir=result_dir, error=error,
                     switch_t=0, min_gain=0.000001)
# table_to_image(norm_data, [num_row, num_col], fea_dist_method, image_dist_method, save_image_size,
#                max_step, val_step, result_dir, error)
