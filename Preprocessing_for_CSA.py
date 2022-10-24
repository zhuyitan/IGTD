import pandas as pd
import sys
import numpy as np
import os
import candle
import _pickle as cp
from Table2Image_Functions import min_max_transform, table_to_image, select_features_by_variation, \
    generate_unique_id_mapping, load_data



fdir = os.path.dirname(os.path.realpath(__file__))

study = ['ccle', 'ctrp', 'gcsi', 'gdsc1', 'gdsc2']

input_data_path = os.path.join(fdir, 'Raw_Data')
output_data_dir = os.path.join(fdir, 'Processed_Data')

candle.get_file(fname='CSA_Data_July2020.tar.gz',
                origin='https://ftp.mcs.anl.gov/pub/candle/public/improve/model_curation_data/IGTD/CSA_Data_July2020.tar.gz',
                unpack=True, md5_hash=None,
                datadir=input_data_path,
                cache_subdir='')

input_data_path = os.path.join(input_data_path, 'CSA_Data_July2020')

if not os.path.exists(output_data_dir):
    os.makedirs(output_data_dir, exist_ok=True)

cancer_col_name = 'CancID'
drug_col_name = 'DrugID'
res_col_name = 'AUC'

# Load all data
data = {}
for s in study:
    data_dir = os.path.join(input_data_path, 'data.' + s)
    data[s] = {}
    data[s]['res'] = pd.read_csv(os.path.join(data_dir, 'rsp_' + s + '.csv'))      # Drug response
    data[s]['ge'] = pd.read_csv(os.path.join(data_dir, 'ge_' + s + '.csv'))        # Gene expressions
    data[s]['ge'].index = data[s]['ge'].iloc[:, 0]
    data[s]['ge'] = data[s]['ge'].iloc[:, 1:]
    data[s]['md'] = pd.read_csv(os.path.join(data_dir, 'mordred_' + s + '.csv'))  # Mordred descriptors
    data[s]['md'].index = data[s]['md'].iloc[:, 0]
    data[s]['md'] = data[s]['md'].iloc[:, 1:]

# Combine all gene expression data and all drug data
ge = data[study[0]]['ge']
md = data[study[0]]['md']
for s in study[1:]:
    if np.sum(ge.columns != data[s]['ge'].columns) != 0:
        sys.exit('Column names of gene expressions do not match')
    if np.sum(md.columns != data[s]['md'].columns) != 0:
        sys.exit('Column names of drug descriptors do not match')
    ge = pd.concat((ge, data[s]['ge']), axis=0)
    md = pd.concat((md, data[s]['md']), axis=0)


# Generate mappings to unique IDs for cancer case IDs and drug IDs
ge_map, ge_unique_data = generate_unique_id_mapping(ge.iloc[:, :1000])
ge_map.columns = ['CancID', 'Unique_CancID']
ge_map.to_csv(os.path.join(output_data_dir, 'CancID_Mapping.txt'), header=True, index=False, sep='\t', line_terminator='\r\n')
ge_unique_data.to_csv(os.path.join(output_data_dir, 'Unique_CancID_Data.txt'), header=True, index=True, sep='\t', line_terminator='\r\n')
md_map, md_unique_data = generate_unique_id_mapping(md.iloc[:, :1000])
md_map.columns = ['DrugID', 'Unique_DrugID']
md_map.to_csv(os.path.join(output_data_dir, 'DrugID_Mapping.txt'), header=True, index=False, sep='\t', line_terminator='\r\n')
md_unique_data.to_csv(os.path.join(output_data_dir, 'Unique_DrugID_Data.txt'), header=True, index=True, sep='\t', line_terminator='\r\n')



# Generate image data of gene expressions with unique IDs
num_row = 20    # Number of pixel rows in image representation
num_col = 20    # Number of pixel columns in image representation
# num_row = 50    # Number of pixel rows in image representation
# num_col = 50    # Number of pixel columns in image representation
num = num_row * num_col # Number of features to be included for analysis, which is also the total number of pixels in image representation
save_image_size = 1 + 16 / 10000 * num  # Size of pictures (in inches) saved during the execution of IGTD algorithm.
max_step = 30000    # The maximum number of iterations to run the IGTD algorithm, if it does not converge.
val_step = 500  # The number of iterations for determining algorithm convergence. If the error reduction rate
# is smaller than a pre-set threshold for val_step itertions, the algorithm converges.

# Import the example data and linearly scale each feature so that its minimum and maximum values are 0 and 1, respectively.
data = pd.read_csv(os.path.join(output_data_dir, 'Unique_CancID_Data.txt'), low_memory=False, sep='\t', engine='c',
                   na_values=['na', '-', ''], header=0, index_col=0)
fid = select_features_by_variation(data, variation_measure='var', threshold=None, num=num)
data = data.iloc[:, fid]
norm_data = min_max_transform(data.values)
norm_data = pd.DataFrame(norm_data, columns=data.columns, index=data.index)

# Run the IGTD algorithm using (1) the Euclidean distance for calculating pairwise feature distances and pariwise pixel
# distances and (2) the absolute function for evaluating the difference between the feature distance ranking matrix and
# the pixel distance ranking matrix. Save the result in Test_1 folder.
fea_dist_method = 'Euclidean'
image_dist_method = 'Euclidean'
error = 'abs'
result_dir = output_data_dir + '/Image_Data/Cancer/'
os.makedirs(name=result_dir, exist_ok=True)
table_to_image(norm_data, [num_row, num_col], fea_dist_method, image_dist_method, save_image_size,
               max_step, val_step, result_dir, error, min_gain=0.000001)



# Generate image data of drug descriptors with unique IDs
num_row = 20    # Number of pixel rows in image representation
num_col = 20    # Number of pixel columns in image representation
# num_row = 37    # Number of pixel rows in image representation
# num_col = 37    # Number of pixel columns in image representation
num = num_row * num_col # Number of features to be included for analysis, which is also the total number of pixels in image representation
save_image_size = 1 + 16 / 10000 * num  # Size of pictures (in inches) saved during the execution of IGTD algorithm.
max_step = 30000    # The maximum number of iterations to run the IGTD algorithm, if it does not converge.
val_step = 500  # The number of iterations for determining algorithm convergence. If the error reduction rate
# is smaller than a pre-set threshold for val_step itertions, the algorithm converges.

# Import the example data and linearly scale each feature so that its minimum and maximum values are 0 and 1, respectively.
data = pd.read_csv(os.path.join(output_data_dir, 'Unique_DrugID_Data.txt'), low_memory=False, sep='\t', engine='c',
                   na_values=['na', '-', ''], header=0, index_col=0)
fid = select_features_by_variation(data, variation_measure='var', threshold=None, num=num)
data = data.iloc[:, fid]
norm_data = min_max_transform(data.values)
norm_data = pd.DataFrame(norm_data, columns=data.columns, index=data.index)

# Run the IGTD algorithm using (1) the Euclidean distance for calculating pairwise feature distances and pariwise pixel
# distances and (2) the absolute function for evaluating the difference between the feature distance ranking matrix and
# the pixel distance ranking matrix. Save the result in Test_1 folder.
fea_dist_method = 'Euclidean'
image_dist_method = 'Euclidean'
error = 'abs'
result_dir = output_data_dir + '/Image_Data/Drug/'
os.makedirs(name=result_dir, exist_ok=True)
table_to_image(norm_data, [num_row, num_col], fea_dist_method, image_dist_method, save_image_size,
               max_step, val_step, result_dir, error, min_gain=0.000001)



# To generate matched gene expression, drug descriptor, and response data
s = 'ccle'
f = 0

cancer_table_data_filepath = os.path.join(input_data_path, 'data.' + s, 'ge_' + s + '.csv')
drug_table_data_filepath = os.path.join(input_data_path, 'data.' + s, 'mordred_' + s + '.csv')
response_data_filepath = os.path.join(input_data_path, 'data.' + s, 'rsp_ccle.csv')
cancer_image_data_filepath = os.path.join(output_data_dir, 'Image_Data', 'Cancer', 'Results.pkl')
drug_image_data_filepath = os.path.join(output_data_dir, 'Image_Data', 'Drug', 'Results.pkl')
cancer_id_mapping_filepath = os.path.join(output_data_dir, 'CancID_Mapping.txt')
drug_id_mapping_filepath = os.path.join(output_data_dir, 'DrugID_Mapping.txt')

# Generate response data of training, validation, and testing sets
res = pd.read_csv(response_data_filepath)

train_id = pd.read_csv(os.path.join(input_data_path, 'data.' + s, 'splits', 'split_' + str(f) + '_tr_id')).values[:, 0]
num_train = len(train_id)
val_id = np.random.permutation(num_train)[:int(np.floor(num_train/9))]
val_id = train_id[val_id]
train_id = np.setdiff1d(train_id, val_id)
test_id = pd.read_csv(os.path.join(input_data_path, 'data.' + s, 'splits', 'split_' + str(f) + '_te_id')).values[:, 0]

train_response_data_filepath = os.path.join(output_data_dir, 'response_train_' + s + '_split_' + str(f) + '.txt')
res.iloc[train_id, :].to_csv(train_response_data_filepath, header=True, index=False, sep='\t', line_terminator='\r\n')
val_response_data_filepath = os.path.join(output_data_dir, 'response_val_' + s + '_split_' + str(f) + '.txt')
res.iloc[val_id, :].to_csv(val_response_data_filepath, header=True, index=False, sep='\t', line_terminator='\r\n')
test_response_data_filepath = os.path.join(output_data_dir, 'response_test_' + s + '_' + s + '_split_' + str(f) + '.txt')
res.iloc[test_id, :].to_csv(test_response_data_filepath, header=True, index=False, sep='\t', line_terminator='\r\n')

# Generate matched data of training set
data = load_data(cancer_table_data_filepath, drug_table_data_filepath, cancer_image_data_filepath,
                 drug_image_data_filepath, cancer_id_mapping_filepath, drug_id_mapping_filepath,
                 cancer_col_name, drug_col_name, res_col_name, response_data_filepath=train_response_data_filepath)
output = open(os.path.join(output_data_dir, 'data_train_' + s + '_split_' + str(f) + '.pkl'), 'wb')
cp.dump(data, output)
output.close()

# Generate matched data of validation set
data = load_data(cancer_table_data_filepath, drug_table_data_filepath, cancer_image_data_filepath,
                 drug_image_data_filepath, cancer_id_mapping_filepath, drug_id_mapping_filepath,
                 cancer_col_name, drug_col_name, res_col_name, response_data_filepath=val_response_data_filepath)
output = open(os.path.join(output_data_dir, 'data_val_' + s + '_split_' + str(f) + '.pkl'), 'wb')
cp.dump(data, output)
output.close()

# Generate matched data of testing sets
data = load_data(cancer_table_data_filepath, drug_table_data_filepath, cancer_image_data_filepath,
                 drug_image_data_filepath, cancer_id_mapping_filepath, drug_id_mapping_filepath,
                 cancer_col_name, drug_col_name, res_col_name, response_data_filepath=[test_response_data_filepath],
                 test_data=[s + '_' + s + '_split_' + str(f)])
output = open(os.path.join(output_data_dir, 'data_test_' + s + '_split_' + str(f) + '.pkl'), 'wb')
cp.dump(data, output)
output.close()
