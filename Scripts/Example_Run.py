import pandas as pd
import numpy as np
import os
import sys
from IGTD_Functions import min_max_transform, table_to_image



num_row = 30
num_col = 30
num = num_row * num_col  # Number of total pixels in image
save_image_size = 3
max_step = 10000
val_step = 300

data = pd.read_csv('../Example_Data/Data.txt', low_memory=False, sep='\t', engine='c', na_values=['na', '-', ''],
                     header=0, index_col=0)
data = data.iloc[:, :num]
norm_data = min_max_transform(data.values)
norm_data = pd.DataFrame(norm_data, columns=data.columns, index=data.index)



fea_dist_method = 'Euclidean'
image_dist_method = 'Euclidean'
error = 'abs'

result_dir = '../Results/Test_1'
os.makedirs(name=result_dir, exist_ok=True)
table_to_image(norm_data, [num_row, num_col], fea_dist_method, image_dist_method, save_image_size,
               max_step, val_step, result_dir, error)



fea_dist_method = 'Pearson'
image_dist_method = 'Manhattan'
error = 'squared'

result_dir = '../Results/Test_2'
os.makedirs(name=result_dir, exist_ok=True)
table_to_image(norm_data, [num_row, num_col], fea_dist_method, image_dist_method, save_image_size,
               max_step, val_step, result_dir, error)
