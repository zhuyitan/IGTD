import numpy as np
import _pickle as cp
import shutil
import os

from Prediction_Modeling_Functions import get_model_parameter, load_example_data, get_data_for_cross_validation, \
    CNN2D_Regression_Analysis, CNN2D_Classification_Analysis



# Load the example drug response data, cell line gene expression image data, and drug descriptor image data
res, ccl, drug = load_example_data()

# Generate sample IDs for 10-fold cross-validation. 8 data folds for training, 1 data fold for validation,
# and 1 data fold for testing
num_fold = 10
num_sample = res.shape[0]
rand_sample_ID = np.random.permutation(num_sample)
fold_size = int(num_sample / num_fold)
sampleID = {}
sampleID['trainID'] = rand_sample_ID[range(fold_size * (num_fold - 2))]
sampleID['valID'] = rand_sample_ID[range(fold_size * (num_fold - 2), fold_size * (num_fold - 1))]
sampleID['testID'] = rand_sample_ID[range(fold_size * (num_fold - 1), num_sample)]



# Run regression prediction
# Create the directory for saving results
result_dir = '../Results/Prediction_On_Images/Regression_Prediction'
if os.path.exists(result_dir):
    shutil.rmtree(result_dir)
os.makedirs(result_dir)

# Load network parameters
para = get_model_parameter('../Data/Example_Model_Parameters/FCNN_Regressor.txt')
subnetwork_para = get_model_parameter('../Data/Example_Model_Parameters/CNN2D_SubNetwork.txt')
para.update(subnetwork_para)

# Generate data for cross-validation analysis
train, val, test = get_data_for_cross_validation(res, ccl, drug, sampleID)

predResult, perM, perf, winningModel, batch_size = CNN2D_Regression_Analysis(train=train, resultFolder=result_dir,
                                                                             para=para, val=val, test=test)

result = {}
result['predResult'] = predResult       # Prediction values for the training, validation, and testing sets
result['perM'] = perM                   # Loss values of training and validation during model training
result['perf'] = perf                   # Prediction performance metrics
result['winningModel'] = winningModel   # Model with the minimum validation loss
result['batch_size'] = batch_size       # Batch size used in model training

# Save prediction performance and all data and results
perf.to_csv(result_dir + '/Prediction_Performance.txt', header=True, index=True, sep='\t', line_terminator='\r\n')
output = open(result_dir + '/Result.pkl', 'wb')
cp.dump(res, output)
cp.dump(ccl, output)
cp.dump(drug, output)
cp.dump(result, output)
cp.dump(sampleID, output)
cp.dump(para, output)
output.close()



# Run classification prediction
# Create the directory for saving results
result_dir = '../Results/Prediction_On_Images/Classification_Prediction'
if os.path.exists(result_dir):
    shutil.rmtree(result_dir)
os.makedirs(result_dir)

# Convert AUC values into response (AUC < 0.5) and non-response (AUC >= 0.5)
id_pos = np.where(res.AUC < 0.5)[0]
id_neg = np.setdiff1d(range(res.shape[0]), id_pos)
res.iloc[id_pos, 2] = 1
res.iloc[id_neg, 2] = 0
res.AUC = res.AUC.astype('int64')

# Load network parameters
para = get_model_parameter('../Data/Example_Model_Parameters/FCNN_Classifier.txt')
subnetwork_para = get_model_parameter('../Data/Example_Model_Parameters/CNN2D_SubNetwork.txt')
para.update(subnetwork_para)

# Generate data for cross-validation analysis
train, val, test = get_data_for_cross_validation(res, ccl, drug, sampleID)

predResult, perM, perf, winningModel, batch_size = CNN2D_Classification_Analysis(train=train, num_class=2,
    resultFolder=result_dir, class_weight='balanced', para=para, val=val, test=test)

result = {}
result['predResult'] = predResult       # Prediction values for the training, validation, and testing sets
result['perM'] = perM                   # Loss values of training and validation during model training
result['perf'] = perf                   # Prediction performance metrics
result['winningModel'] = winningModel   # Model with the minimum validation loss
result['batch_size'] = batch_size       # Batch size used in model training

# Save prediction performance and all data and results
perf.to_csv(result_dir + '/Prediction_Performance.txt', header=True, index=True, sep='\t', line_terminator='\r\n')
output = open(result_dir + '/Result.pkl', 'wb')
cp.dump(res, output)
cp.dump(ccl, output)
cp.dump(drug, output)
cp.dump(result, output)
cp.dump(sampleID, output)
cp.dump(para, output)
output.close()