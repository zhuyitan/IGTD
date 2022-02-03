from keras import backend
from keras import optimizers
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, concatenate, Conv2D, BatchNormalization, ReLU, MaxPooling2D, \
    Flatten, AlphaDropout
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, roc_auc_score, accuracy_score, \
    matthews_corrcoef

import configparser
import numpy as np
import keras
import os
import pandas as pd
import shutil



def ID_mapping(l1, l2):
    pos = {}
    for i in range(len(l1)):
        pos[l1[i]] = i
    idd = np.array([pos[i] for i in l2])
    return idd



def load_example_data():
    res = pd.read_csv('../Data/Example_Drug_Response_Data.txt', sep='\t', engine='c',
                      na_values=['na', '-', ''], header=0, index_col=None)

    files = os.listdir('../Data/Example_Drug_Descriptor_Image_Data/')
    image = np.empty((len(files), 50, 50, 1))
    sample = []
    id = []
    for i in range(len(files)):
        if files[i].split('.')[1] == 'txt' and files[i].split('_')[0] == 'Drug':
            id.append(i)
            data = pd.read_csv('../Data/Example_Drug_Descriptor_Image_Data/' + files[i], sep='\t', engine='c',
                               na_values=['na', '-', ''], header=None, index_col=None)
            image[i, :, :, 0] = data.values
            sample.append(files[i].split('.txt')[0])
    image = image[id, :, :, :]
    drug = {}
    drug['data'] = image
    drug['sample'] = sample

    files = os.listdir('../Data/Example_Gene_Expression_Image_Data/')
    image = np.empty((len(files), 50, 50, 1))
    sample = []
    id = []
    for i in range(len(files)):
        if files[i].split('.')[1] == 'txt' and files[i].split('_')[0] == 'CCL':
            id.append(i)
            data = pd.read_csv('../Data/Example_Gene_Expression_Image_Data/' + files[i], sep='\t', engine='c',
                               na_values=['na', '-', ''], header=None, index_col=None)
            image[i, :, :, 0] = data.values
            sample.append(files[i].split('.txt')[0])
    image = image[id, :, :, :]
    ccl = {}
    ccl['data'] = image
    ccl['sample'] = sample

    return res, ccl, drug



def get_data_for_cross_validation(res, ccl, drug, sampleID):

    trainData = []
    valData = []
    testData = []

    train_idd = ID_mapping(drug['sample'], res.iloc[sampleID['trainID'], :].Drug)
    trainData.append(drug['data'][train_idd, :, :, :])
    val_idd = ID_mapping(drug['sample'], res.iloc[sampleID['valID'], :].Drug)
    valData.append(drug['data'][val_idd, :, :, :])
    test_idd = ID_mapping(drug['sample'], res.iloc[sampleID['testID'], :].Drug)
    testData.append(drug['data'][test_idd, :, :, :])

    train_idd = ID_mapping(ccl['sample'], res.iloc[sampleID['trainID'], :].CCL)
    trainData.append(ccl['data'][train_idd, :, :, :])
    val_idd = ID_mapping(ccl['sample'], res.iloc[sampleID['valID'], :].CCL)
    valData.append(ccl['data'][val_idd, :, :, :])
    test_idd = ID_mapping(ccl['sample'], res.iloc[sampleID['testID'], :].CCL)
    testData.append(ccl['data'][test_idd, :, :, :])

    trainLabel = res.iloc[sampleID['trainID'], :].AUC.values
    valLabel = res.iloc[sampleID['valID'], :].AUC.values
    testLabel = res.iloc[sampleID['testID'], :].AUC.values

    train = {}
    train['data'] = trainData
    train['label'] = trainLabel
    train['sample'] = res.iloc[sampleID['trainID'], :].CCL + '|' + res.iloc[sampleID['trainID'], :].Drug
    val = {}
    val['data'] = valData
    val['label'] = valLabel
    val['sample'] = res.iloc[sampleID['valID'], :].CCL + '|' + res.iloc[sampleID['valID'], :].Drug
    test = {}
    test['data'] = testData
    test['label'] = testLabel
    test['sample'] = res.iloc[sampleID['testID'], :].CCL + '|' + res.iloc[sampleID['testID'], :].Drug

    return train, val, test



def get_model_parameter(model_file):
    config = configparser.ConfigParser()
    config.read(model_file)
    section = config.sections()
    params = {}
    for sec in section:
        for k, v in config.items(sec):
            if k not in params:
                params[k] = eval(v)
    return params



def get_DNN_optimizer(opt_name):
    if opt_name == 'SGD':
        optimizer = optimizers.GSD()
    elif opt_name == 'SGD_momentum':
        optimizer = optimizers.GSD(momentum=0.9)
    elif opt_name == 'SGD_momentum_nesterov':
        optimizer = optimizers.GSD(momentum=0.9, nesterov=True)
    elif opt_name == 'RMSprop':
        optimizer = optimizers.RMSprop()
    elif opt_name == 'Adagrad':
        optimizer = optimizers.Adagrad()
    elif opt_name == 'Adadelta':
        optimizer = optimizers.Adadelta()
    elif opt_name == 'Adam':
        optimizer = optimizers.Adam()
    elif opt_name == 'Adam_amsgrad':
        optimizer = optimizers.Adam(amsgrad=True)
    else:
        optimizer = optimizers.Adam()

    return optimizer



def calculate_batch_size(num_sample, paraDNN):
    # max_half_num_batch: the number of batches will not be larger than 2 * max_half_num_batch
    max_half_num_batch = paraDNN['max_half_num_batch']
    if num_sample < max_half_num_batch * 4:
        batch_size = 2
    elif num_sample < max_half_num_batch * 8:
        batch_size = 4
    elif num_sample < max_half_num_batch * 16:
        batch_size = 8
    elif num_sample < max_half_num_batch * 32:
        batch_size = 16
    elif num_sample < max_half_num_batch * 64:
        batch_size = 32
    elif num_sample < max_half_num_batch * 128:
        batch_size = 64
    elif num_sample < max_half_num_batch * 256:
        batch_size = 128
    else:
        batch_size = 256

    return batch_size



class CNN2D_Regressor():
    # This is the class for 2-dimensional convolutional neural network regressors (CNN2D_Regressor).
    # The model can accept more than one sizes of filter, such [3, 5].
    def __init__(self, params, input_data_dim, dropout):
        # params: dict, CNN2D model parameters
        # input_data_dim: a list. Each element of the list includes two positive integers, the dimension of input images
        # dropout: dropout rate, all layers use the same dropout rate

        self.params = params
        self.dropout = dropout
        self.input_data_dim = input_data_dim

        num_kernel_size = len(self.params['kernel_size'])
        num_conv_layer = []
        for i in range(num_kernel_size):
            num_conv_layer.append(len(self.params['num_kernel'][i]))
        num_dense_layer = len(self.params['network_layers'])

        input = []
        input2List = []
        num_input = len(self.input_data_dim)
        for input_id in range(num_input):
            in_id = Input(shape=(self.input_data_dim[input_id][0], self.input_data_dim[input_id][1], 1),
                          name='Input_' + str(input_id))
            input.append(in_id)
            for j in range(num_kernel_size):
                min_row_size = self.params['pool_size'][j][0] * 2 + self.params['kernel_size'][j][0] - 1
                min_col_size = self.params['pool_size'][j][1] * 2 + self.params['kernel_size'][j][1] - 1
                for i in range(num_conv_layer[j]):
                    if i == 0:
                        d = Conv2D(filters=self.params['num_kernel'][j][i], kernel_size=self.params['kernel_size'][j],
                                   strides=self.params['strides'][j], padding='valid', data_format='channels_last',
                                   name='Conv2D_' + str(i) + '_Kernel_' + str(j) + '_Input_' + str(input_id))(in_id)
                    else:
                        d = Conv2D(filters=self.params['num_kernel'][j][i], kernel_size=self.params['kernel_size'][j],
                                   strides=self.params['strides'][j], padding='valid', data_format='channels_last',
                                   name='Conv2D_' + str(i) + '_Kernel_' + str(j) + '_Input_' + str(input_id))(d)
                    d = BatchNormalization(axis=-1, name='BatchNorm_' + str(i) + '_Kernel_' + str(j) + '_Input_'
                                                         + str(input_id))(inputs=d)
                    if self.params['subnetwork_activation'] == 'relu':
                        d = ReLU(name='ReLU_' + str(i) + '_Kernel_' + str(j) + '_Input_' + str(input_id))(d)
                    else:
                        raise TypeError("Activation is not ReLU in subnetwork.")
                    d = MaxPooling2D(pool_size=self.params['pool_size'][j], name='MaxPooling_' + str(i) + '_Kernel_'
                        + str(j) + '_Input_' + str(input_id))(d)
                    dim = np.array(d.shape.as_list())
                    flag_0 = dim[1] < min_row_size
                    flag_1 = dim[2] < min_col_size
                    if flag_0 or flag_1:
                        break
                d = Flatten()(d)
                input2List.append(d)

        if num_input > 1:
            d = concatenate(input2List, name='concatenation')
        for i in range(num_dense_layer):
            if self.params['activation'] == 'selu':
                d = Dense(self.params['network_layers'][i], activation=self.params['activation'], name='Dense_' + str(i),
                          kernel_initializer='lecun_normal')(d)
            else:
                d = Dense(self.params['network_layers'][i], activation=self.params['activation'], name='Dense_' + str(i))(d)
            if i != num_dense_layer - 1:
                if self.params['activation'] == 'selu':
                    d = AlphaDropout(self.dropout, name='Dropout_Dense_' + str(i))(d)
                else:
                    d = Dropout(self.dropout, name='Dropout_Dense_' + str(i))(d)

        output = Dense(1, name='output')(d)
        if num_input > 1:
            model = Model(inputs=input, outputs=output)
        else:
            model = Model(inputs=input[0], outputs=output)
        model.compile(optimizer=get_DNN_optimizer(self.params['optimizer']), loss=self.params['loss'])
        print(model.summary())
        self.model = model



class CNN2D_Classifier():
    # This is the class for 2-dimensional convolutional neural network regressors (CNN2D_Regressor).
    # The model can accept more than one sizes of filter, such [3, 5].
    def __init__(self, params, input_data_dim, num_class, dropout):
        # params: dict, CNN2D model parameters
        # input_data_dim: a list. Each element of the list includes two positive integers, the dimension of input images
        # dropout: dropout rate, all layers use the same dropout rate

        self.params = params
        self.dropout = dropout
        self.num_class = num_class
        self.input_data_dim = input_data_dim

        num_kernel_size = len(self.params['kernel_size'])
        num_conv_layer = []
        for i in range(num_kernel_size):
            num_conv_layer.append(len(self.params['num_kernel'][i]))
        num_dense_layer = len(self.params['network_layers'])

        input = []
        input2List = []
        num_input = len(self.input_data_dim)
        for input_id in range(num_input):
            in_id = Input(shape=(self.input_data_dim[input_id][0], self.input_data_dim[input_id][1], 1),
                          name='Input_' + str(input_id))
            input.append(in_id)
            for j in range(num_kernel_size):
                min_row_size = self.params['pool_size'][j][0] * 2 + self.params['kernel_size'][j][0] - 1
                min_col_size = self.params['pool_size'][j][1] * 2 + self.params['kernel_size'][j][1] - 1
                for i in range(num_conv_layer[j]):
                    if i == 0:
                        d = Conv2D(filters=self.params['num_kernel'][j][i], kernel_size=self.params['kernel_size'][j],
                                   strides=self.params['strides'][j], padding='valid', data_format='channels_last',
                                   name='Conv2D_' + str(i) + '_Kernel_' + str(j) + '_Input_' + str(input_id))(in_id)
                    else:
                        d = Conv2D(filters=self.params['num_kernel'][j][i], kernel_size=self.params['kernel_size'][j],
                                   strides=self.params['strides'][j], padding='valid', data_format='channels_last',
                                   name='Conv2D_' + str(i) + '_Kernel_' + str(j) + '_Input_' + str(input_id))(d)
                    d = BatchNormalization(axis=-1, name='BatchNorm_' + str(i) + '_Kernel_' + str(j) + '_Input_'
                                                         + str(input_id))(inputs=d)
                    if self.params['subnetwork_activation'] == 'relu':
                        d = ReLU(name='ReLU_' + str(i) + '_Kernel_' + str(j) + '_Input_' + str(input_id))(d)
                    else:
                        raise TypeError("Activation is not ReLU in subnetwork.")
                    d = MaxPooling2D(pool_size=self.params['pool_size'][j], name='MaxPooling_' + str(i) + '_Kernel_'
                        + str(j) + '_Input_' + str(input_id), padding='same')(d)
                    dim = np.array(d.shape.as_list())
                    flag_0 = dim[1] < min_row_size
                    flag_1 = dim[2] < min_col_size
                    if flag_0 or flag_1:
                        break
                d = Flatten()(d)
                input2List.append(d)

        if num_input > 1:
            d = concatenate(input2List, name='concatenation')
        for i in range(num_dense_layer):
            if self.params['activation'] == 'selu':
                d = Dense(self.params['network_layers'][i], activation=self.params['activation'], name='Dense_' + str(i),
                          kernel_initializer='lecun_normal')(d)
            else:
                d = Dense(self.params['network_layers'][i], activation=self.params['activation'], name='Dense_' + str(i))(d)
            if i != num_dense_layer - 1:
                if self.params['activation'] == 'selu':
                    d = AlphaDropout(self.dropout, name='Dropout_Dense_' + str(i))(d)
                else:
                    d = Dropout(self.dropout, name='Dropout_Dense_' + str(i))(d)

        output = Dense(self.num_class, activation='softmax', name='output')(d)
        if num_input > 1:
            model = Model(inputs=input, outputs=output)
        else:
            model = Model(inputs=input[0], outputs=output)
        model.compile(optimizer=get_DNN_optimizer(self.params['optimizer']), loss=self.params['loss'])
        print(model.summary())
        self.model = model



def CNN2D_Regression_Analysis(train, resultFolder, para, val=None, test=None):
    '''
    This function does CNN2D regression analysis without HPO.

    Input:
    train: a dictionary of three elements. data is an array of (sample, height, width).
        label is a series of the prediction target. sample is an array of sample names.
    val: a dictionary for validation data.
    resultFolder: directory to save models, features, and results
    para: parameters used for model training
    test: a dictionary for testing data. Default is None.

    Return:
    predResult: a dictionary including three series, which are prediction results on the training, validation,
        and testing sets.
    perM: an array of training and validation losses with different dropout rates and epochs.
    perf: a 3 by 7 data frame including the prediction performance on training, validation, and testing sets.
    winningModel: a string giving the epoch number and dropout rate of the best model with the smallest validation loss.
    '''

    if os.path.exists(resultFolder):
        shutil.rmtree(resultFolder)
    os.mkdir(resultFolder)

    trainData = train['data']
    trainLabel = train['label']
    trainSample = train['sample']

    if isinstance(trainData, list):
        batch_size = calculate_batch_size(trainData[0].shape[0], para)
    else:
        batch_size = calculate_batch_size(trainData.shape[0], para)

    # batch_size = 5000
    # print(batch_size)

    if val is not None:
        valData = val['data']
        valLabel = val['label']
        valSample = val['sample']
    else:
        valData = None
        valLabel = None
        valSample = None

    if test is not None:
        testData = test['data']
        testSample = test['sample']
        if test['label'] is not None:
            testLabel = test['label']
        else:
            testLabel = None
    else:
        testData = None
        testLabel = None
        testSample = None

    if isinstance(trainData, list):
        input_data_dim = []
        for i in range(len(trainData)):
            input_data_dim.append([trainData[i].shape[1], trainData[i].shape[2]])
    else:
        input_data_dim = [[trainData.shape[1], trainData.shape[2]]]

    perM = {}
    for i in ['train', 'val']:
        perM[i] = np.empty((len(para['drop']), para['epochs']))
        perM[i].fill(np.inf)
        perM[i] = pd.DataFrame(perM[i], index=['dropout_' + str(j) for j in para['drop']],
            columns=['epoch_' + str(j) for j in range(para['epochs'])])

    for dpID in range(len(para['drop'])):
        label = 'dropout_' + str(para['drop'][dpID])
        print(label)

        if val is not None:
            monitor = 'val_loss'
        else:
            monitor = 'loss'
        train_logger = CSVLogger(resultFolder + '/log_dropout_' + str(para['drop'][dpID]) + '.csv')
        model_saver = ModelCheckpoint(resultFolder + '/model_dropout_' + str(para['drop'][dpID]) + '.h5',
                                      monitor=monitor, save_best_only=True, save_weights_only=False)
        reduce_lr = ReduceLROnPlateau(monitor=monitor, factor=para['rlr_factor'], patience=para['rlr_patience'],
                                      verbose=1, mode='auto', min_delta=para['rlr_min_delta'],
                                      cooldown=para['rlr_cooldown'], min_lr=para['rlr_min_lr'])
        early_stop = EarlyStopping(monitor=monitor, patience=para['es_patience'], min_delta=para['es_min_delta'],
                                   verbose=1)
        callbacks = [model_saver, train_logger, reduce_lr, early_stop]

        temp = CNN2D_Regressor(para, input_data_dim, para['drop'][dpID])

        if val is not None:
            history = temp.model.fit(x=trainData, y=trainLabel, batch_size=batch_size, epochs=para['epochs'],
                verbose=para['verbose'], callbacks=callbacks, validation_data=(valData, valLabel), shuffle=True)
        else:
            history = temp.model.fit(x=trainData, y=trainLabel, batch_size=batch_size, epochs=para['epochs'],
                verbose=para['verbose'], callbacks=callbacks, validation_data=None, shuffle=True)
        numEpoch = len(history.history['loss'])
        i = np.where(perM['train'].index == label)[0]
        perM['train'].iloc[i, :numEpoch] = history.history['loss']
        if val is not None:
            numEpoch = len(history.history['val_loss'])
            i = np.where(perM['val'].index == label)[0]
            perM['val'].iloc[i, :numEpoch] = history.history['val_loss']

        backend.clear_session()

    if val is not None:
        dpID, epID = np.unravel_index(np.argmin(perM['val'].values, axis=None), perM['val'].shape)
    else:
        dpID, epID = np.unravel_index(np.argmin(perM['train'].values, axis=None), perM['train'].shape)
    model = load_model(resultFolder + '/model_dropout_' + str(para['drop'][dpID]) + '.h5')

    for i in range(len(para['drop'])):
        if i == dpID:
            continue
        os.remove(resultFolder + '/model_dropout_' + str(para['drop'][i]) + '.h5')
        os.remove(resultFolder + '/log_dropout_' + str(para['drop'][i]) + '.csv')

    predResult = {}
    if test is not None:
        predResult['test'] = pd.DataFrame(model.predict(testData), index=testSample, columns=['prediction'])
    predResult['train'] = pd.DataFrame(model.predict(trainData), index=trainSample, columns=['prediction'])
    if val is not None:
        predResult['val'] = pd.DataFrame(model.predict(valData), index=valSample, columns=['prediction'])

    backend.clear_session()

    perf = np.empty((3, 7))
    perf.fill(np.nan)
    perf = pd.DataFrame(perf, columns=['R2', 'MSE', 'MAE', 'pCor', 'pCorPvalue', 'sCor', 'sCorPvalue'],
                        index=['train', 'val', 'test'])
    for k in ['train', 'val', 'test']:
        if (eval(k + 'Data') is None) or (eval(k + 'Label') is None):
            continue
        perf.loc[k, 'R2'] = r2_score(eval(k + 'Label'), predResult[k].values[:, 0])
        perf.loc[k, 'MSE'] = mean_squared_error(eval(k + 'Label'), predResult[k].values[:, 0])
        perf.loc[k, 'MAE'] = mean_absolute_error(eval(k + 'Label'), predResult[k].values[:, 0])
        rho, pval = stats.pearsonr(eval(k + 'Label'), predResult[k].values[:, 0])
        perf.loc[k, 'pCor'] = rho
        perf.loc[k, 'pCorPvalue'] = pval
        rho, pval = stats.spearmanr(eval(k + 'Label'), predResult[k].values[:, 0])
        perf.loc[k, 'sCor'] = rho
        perf.loc[k, 'sCorPvalue'] = pval

    return predResult, perM, perf, 'dropout_' + str(para['drop'][dpID]) + '_epoch_' + str(epID + 1), batch_size



def CNN2D_Classification_Analysis(train, num_class, resultFolder, para, class_weight=None, val=None, test=None):
    '''
    This function does CNN2D regression analysis without HPO.

    Input:
    train: a dictionary of three elements. data is an array of (sample, height, width).
        label is a series of the prediction target. sample is an array of sample names.
    val: a dictionary for validation data.
    resultFolder: directory to save models, features, and results
    para: parameters used for model training
    test: a dictionary for testing data. Default is None.

    Return:
    predResult: a dictionary including three series, which are prediction results on the training, validation,
        and testing sets.
    perM: an array of training and validation losses with different dropout rates and epochs.
    perf: a 3 by 7 data frame including the prediction performance on training, validation, and testing sets.
    winningModel: a string giving the epoch number and dropout rate of the best model with the smallest validation loss.
    '''

    if os.path.exists(resultFolder):
        shutil.rmtree(resultFolder)
    os.mkdir(resultFolder)

    trainData = train['data']
    trainLabel = train['label']
    trainSample = train['sample']

    if isinstance(trainData, list):
        batch_size = calculate_batch_size(trainData[0].shape[0], para)
    else:
        batch_size = calculate_batch_size(trainData.shape[0], para)
    # batch_size = 5000
    # print(batch_size)

    if val is not None:
        valData = val['data']
        valLabel = val['label']
        valSample = val['sample']
    else:
        valData = None
        valLabel = None
        valSample = None

    if test is not None:
        testData = test['data']
        testSample = test['sample']
        if test['label'] is not None:
            testLabel = test['label']
        else:
            testLabel = None
    else:
        testData = None
        testLabel = None
        testSample = None


    if isinstance(trainData, list):
        input_data_dim = []
        for i in range(len(trainData)):
            input_data_dim.append([trainData[i].shape[1], trainData[i].shape[2]])
    else:
        input_data_dim = [[trainData.shape[1], trainData.shape[2]]]

    perM = {}
    for i in ['train', 'val']:
        perM[i] = np.empty((len(para['drop']), para['epochs']))
        perM[i].fill(np.inf)
        perM[i] = pd.DataFrame(perM[i], index=['dropout_' + str(j) for j in para['drop']],
            columns=['epoch_' + str(j) for j in range(para['epochs'])])

    if class_weight == 'balanced':
        weight = len(trainLabel) / (num_class * np.bincount(trainLabel))
        class_weight = {}
        for i in range(num_class):
            class_weight[i] = weight[i]

    for dpID in range(len(para['drop'])):
        label = 'dropout_' + str(para['drop'][dpID])
        print(label)

        if val is not None:
            monitor = 'val_loss'
        else:
            monitor = 'loss'
        train_logger = CSVLogger(resultFolder + '/log_dropout_' + str(para['drop'][dpID]) + '.csv')
        model_saver = ModelCheckpoint(resultFolder + '/model_dropout_' + str(para['drop'][dpID]) + '.h5',
                                      monitor=monitor, save_best_only=True, save_weights_only=False)
        reduce_lr = ReduceLROnPlateau(monitor=monitor, factor=para['rlr_factor'], patience=para['rlr_patience'],
                                      verbose=1, mode='auto', min_delta=para['rlr_min_delta'],
                                      cooldown=para['rlr_cooldown'], min_lr=para['rlr_min_lr'])
        early_stop = EarlyStopping(monitor=monitor, patience=para['es_patience'], min_delta=para['es_min_delta'],
                                   verbose=1)
        callbacks = [model_saver, train_logger, reduce_lr, early_stop]

        temp = CNN2D_Classifier(para, input_data_dim, num_class, para['drop'][dpID])

        if val is not None:
            history = temp.model.fit(x=trainData, y=trainLabel, batch_size=batch_size, epochs=para['epochs'],
                verbose=para['verbose'], callbacks=callbacks, validation_data=(valData, valLabel),
                class_weight=class_weight, shuffle=True)
        else:
            history = temp.model.fit(x=trainData, y=trainLabel, batch_size=batch_size, epochs=para['epochs'],
                verbose=para['verbose'], callbacks=callbacks, validation_data=None, class_weight=class_weight,
                shuffle=True)
        numEpoch = len(history.history['loss'])
        i = np.where(perM['train'].index == label)[0]
        perM['train'].iloc[i, :numEpoch] = history.history['loss']
        if val is not None:
            numEpoch = len(history.history['val_loss'])
            i = np.where(perM['val'].index == label)[0]
            perM['val'].iloc[i, :numEpoch] = history.history['val_loss']

        backend.clear_session()

    if val is not None:
        dpID, epID = np.unravel_index(np.argmin(perM['val'].values, axis=None), perM['val'].shape)
    else:
        dpID, epID = np.unravel_index(np.argmin(perM['train'].values, axis=None), perM['train'].shape)
    model = load_model(resultFolder + '/model_dropout_' + str(para['drop'][dpID]) + '.h5')

    for i in range(len(para['drop'])):
        if i == dpID:
            continue
        os.remove(resultFolder + '/model_dropout_' + str(para['drop'][i]) + '.h5')
        os.remove(resultFolder + '/log_dropout_' + str(para['drop'][i]) + '.csv')

    predResult = {}
    if test is not None:
        predResult['test'] = {}
        predResult['test']['proba'] = pd.DataFrame(model.predict(testData), index=testSample,
                                                   columns=['proba_' + str(i) for i in range(num_class)])
        predResult['test']['label'] = pd.DataFrame(np.argmax(a=predResult['test']['proba'].values, axis=1),
                                                   index=predResult['test']['proba'].index, columns=['prediction'])
    predResult['train'] = {}
    predResult['train']['proba'] = pd.DataFrame(model.predict(trainData), index=trainSample,
                                               columns=['proba_' + str(i) for i in range(num_class)])
    predResult['train']['label'] = pd.DataFrame(np.argmax(a=predResult['train']['proba'].values, axis=1),
                                               index=predResult['train']['proba'].index, columns=['prediction'])
    if val is not None:
        predResult['val'] = {}
        predResult['val']['proba'] = pd.DataFrame(model.predict(valData), index=valSample,
                                                  columns=['proba_' + str(i) for i in range(num_class)])
        predResult['val']['label'] = pd.DataFrame(np.argmax(a=predResult['val']['proba'].values, axis=1),
                                                  index=predResult['val']['proba'].index, columns=['prediction'])

    backend.clear_session()

    perf = np.empty((3, 3))
    perf.fill(np.nan)
    perf = pd.DataFrame(perf, columns=['ACC', 'AUROC', 'MCC'], index=['train', 'val', 'test'])
    for k in ['train', 'val', 'test']:
        if (eval(k + 'Data') is None) or (eval(k + 'Label') is None):
            continue
        perf.loc[k, 'ACC'] = accuracy_score(eval(k + 'Label'), predResult[k]['label'].values[:, 0])
        if num_class == 2:
            perf.loc[k, 'AUROC'] = roc_auc_score(eval(k + 'Label'), predResult[k]['proba'].values[:, 1])
        else:
            perf.loc[k, 'AUROC'] = roc_auc_score(keras.utils.to_categorical(eval(k + 'Label')), predResult[k]['proba'].values,
                                                 labels=range(num_class), multi_class='ovr')
        perf.loc[k, 'MCC'] = matthews_corrcoef(eval(k + 'Label'), predResult[k]['label'].values[:, 0])

    return predResult, perM, perf, 'dropout_' + str(para['drop'][dpID]) + '_epoch_' + str(epID + 1), batch_size
