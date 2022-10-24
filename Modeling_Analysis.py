import candle
import os
import numpy as np
import pandas as pd
import _pickle as cp
from Modeling_Functions import CNN2D_Regressor, CNN2D_Classifier
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from keras import backend
from keras.models import load_model
from tensorflow.keras import backend as K





# This should be set outside as a user environment variable
file_path = os.path.dirname(os.path.realpath(__file__))


additional_definitions = [
    {'name': 'cancer_col_name',
     'type': str,
     'help': 'Column name of cancer ID'
     },
    {'name': 'drug_col_name',
     'type': str,
     'help': 'Column name of drug ID'
     },
    {'name': 'res_col_name',
     'type': str,
     'help': 'Column name of response'
     },
    {'name': 'rlr_factor',
     'type': float,
     'help': 'Learning rate reduction factor'
     },
    {'name': 'rlr_min_delta',
     'type': float,
     'help': 'Learning rate reduction minimum delta'
     },
    {'name': 'rlr_cooldown',
     'type': int,
     'help': 'Learning rate reduction cooldown'
     },
    {'name': 'rlr_min_lr',
     'type': float,
     'help': 'Learning rate reduction minimum learning rate'
     },
    {'name': 'rlr_patience',
     'type': int,
     'help': 'Learning rate reduction patience'
     },
    {'name': 'es_patience',
     'type': int,
     'help': 'Early stop patience'
     },
    {'name': 'es_min_delta',
     'type': float,
     'help': 'Early stop minimum delta'
     },
    {'name': 'classification_task',
     'type': bool,
     'help': 'Is the task classification or not'
     },
    {'name': 'cnn_activation',
     'type': str,
     'help': 'Activation function for convolution layers'
     }
]

required = ['output_dir', 'conv', 'dropout', 'epochs', 'pool', 'dense', 'activation', 'loss',
            'optimizer', 'verbose', 'batch_size', 'early_stop', 'train_data', 'val_data', 'test_data',
            'data_url']


# experimental
# supported_definitions = ['data_url', 'shuffle', 'feature_subsample']


class igtd(candle.Benchmark):

    def set_locals(self):
        if required is not None:
            self.required = set(required)
        if additional_definitions is not None:
            self.additional_definitions = additional_definitions


def initialize_parameters():
    igtd_bmk = igtd(file_path, 'Default_Params.txt', 'keras', prog='igtd', desc='Candle compliant IGTD')
    gParameters = candle.finalize_parameters(igtd_bmk)

    return gParameters



def run(params):
# fetch data
# preprocess data
# save preprocessed data
# define callbacks
# build / compile model
# train model
# infer using model
# etc

    os.environ['CANDLE_DATA_DIR'] = params['candle_data_dir']
    params['verbose'] = 2

    candle.get_file(fname=params['train_data'],
                origin=os.path.join(params['data_url'], params['train_data']),
                unpack=False, md5_hash=None,
                datadir=None,
                cache_subdir='')
    pkl_file = open(os.path.join(params['candle_data_dir'], params['train_data']), 'rb')
    temp_data = cp.load(pkl_file)
    pkl_file.close()
    trainData = temp_data['data']
    trainLabel = temp_data['label']
    trainSample = temp_data['sample']

    if 'val_data' in params.keys():
        candle.get_file(fname=params['val_data'],
                        origin=os.path.join(params['data_url'], params['val_data']),
                        unpack=False, md5_hash=None,
                        datadir=None,
                        cache_subdir='')
        pkl_file = open(os.path.join(params['candle_data_dir'], params['val_data']), 'rb')
        temp_data = cp.load(pkl_file)
        pkl_file.close()
        valData = temp_data['data']
        valLabel = temp_data['label']
        valSample = temp_data['sample']
        monitor = 'val_loss'
    else:
        valData = None
        valLabel = None
        valSample = None
        monitor = 'loss'

    if 'test_data' in params.keys():
        candle.get_file(fname=params['test_data'],
                        origin=os.path.join(params['data_url'], params['test_data']),
                        unpack=False, md5_hash=None,
                        datadir=None,
                        cache_subdir='')
        pkl_file = open(os.path.join(params['candle_data_dir'], params['test_data']), 'rb')
        temp_data = cp.load(pkl_file)
        pkl_file.close()
        testData = temp_data['data']
        testLabel = temp_data['label']
        testSample = temp_data['sample']
    else:
        testData = None
        testLabel = None
        testSample = None

    batch_size = params['batch_size']

    if isinstance(trainData, list):
        input_data_dim = []
        for i in range(len(trainData)):
            input_data_dim.append([trainData[i].shape[1], trainData[i].shape[2]])
    else:
        input_data_dim = [[trainData.shape[1], trainData.shape[2]]]

    train_logger = CSVLogger(params['output_dir'] + '/log.csv')
    model_saver = ModelCheckpoint(params['output_dir'] + '/model.h5',
                                  monitor=monitor, save_best_only=True, save_weights_only=False)
    reduce_lr = ReduceLROnPlateau(monitor=monitor, factor=params['rlr_factor'], patience=params['rlr_patience'],
                                  verbose=1, mode='auto', min_delta=params['rlr_min_delta'],
                                  cooldown=params['rlr_cooldown'], min_lr=params['rlr_min_lr'])
    early_stop = EarlyStopping(monitor=monitor, patience=params['es_patience'], min_delta=params['es_min_delta'],
                               verbose=1)
    callbacks = [model_saver, train_logger, reduce_lr, early_stop]

    if params['classification_task']:
        num_class = int(len(np.unique(trainLabel)))
        temp = CNN2D_Classifier(params, input_data_dim, num_class, params['dropout'])
        weight = len(trainLabel) / (num_class * np.bincount(trainLabel))
        class_weight = {}
        for i in range(num_class):
            class_weight[i] = weight[i]
    else:
        temp = CNN2D_Regressor(params, input_data_dim, params['dropout'])

    if valData is None:
        if params['classification_task']:
            history = temp.model.fit(x=trainData, y=trainLabel, batch_size=batch_size, epochs=params['epochs'],
                                     verbose=params['verbose'], callbacks=callbacks, validation_data=None,
                                     class_weight=class_weight, shuffle=True)
        else:
            history = temp.model.fit(x=trainData, y=trainLabel, batch_size=batch_size, epochs=params['epochs'],
                                     verbose=params['verbose'], callbacks=callbacks, validation_data=None,
                                     shuffle=True)
    else:
        if params['classification_task']:
            history = temp.model.fit(x=trainData, y=trainLabel, batch_size=batch_size, epochs=params['epochs'],
                                     verbose=params['verbose'], callbacks=callbacks, validation_data=(valData, valLabel),
                                     class_weight=class_weight, shuffle=True)
        else:
            history = temp.model.fit(x=trainData, y=trainLabel, batch_size=batch_size, epochs=params['epochs'],
                                     verbose=params['verbose'], callbacks=callbacks, validation_data=(valData, valLabel),
                                     shuffle=True)
    backend.clear_session()
    model = load_model(params['output_dir'] + '/model.h5')
    predResult = {}
    if testData is not None:
        predResult['test'] = {}
        for s in testData.keys():
            if params['classification_task']:
                testPredResult = np.argmax(a=model.predict(testData[s]).values, axis=1)
            else:
                testPredResult = model.predict(testData[s])[:, 0]
            if testLabel[s] is not None:
                predResult['test'][s] = pd.DataFrame({params['cancer_col_name']: [i.split('|')[0] for i in testSample[s]],
                                                   params['drug_col_name']: [i.split('|')[1] for i in testSample[s]],
                                                   params['res_col_name']: testLabel[s],
                                                   'Prediction': testPredResult}, index=testSample[s])
            else:
                predResult['test'][s] = pd.DataFrame({params['cancer_col_name']: [i.split('|')[0] for i in testSample[s]],
                                                   params['drug_col_name']: [i.split('|')[1] for i in testSample[s]],
                                                   'Prediction': testPredResult}, index=testSample[s])
            predResult['test'][s].to_csv(params['output_dir'] + '/' + 'Test_' + s + '_Prediction_Result.txt',
                                         header=True, index=False, sep='\t', line_terminator='\r\n')

    if params['classification_task']:
        trainPredResult = np.argmax(a=model.predict(trainData).values, axis=1)
    else:
        trainPredResult = model.predict(trainData)[:, 0]
    predResult['train'] = pd.DataFrame({params['cancer_col_name']: [i.split('|')[0] for i in trainSample],
                                        params['drug_col_name']: [i.split('|')[1] for i in trainSample],
                                        params['res_col_name']: trainLabel,
                                        'Prediction': trainPredResult}, index=trainSample)
    predResult['train'].to_csv(params['output_dir'] + '/' + 'Train_Prediction_Result.txt', header=True,
                               index=False, sep='\t', line_terminator='\r\n')

    if valData is not None:
        if params['classification_task']:
            valPredResult = np.argmax(a=model.predict(valData).values, axis=1)
        else:
            valPredResult = model.predict(valData)[:, 0]
        predResult['val'] = pd.DataFrame({params['cancer_col_name']: [i.split('|')[0] for i in valSample],
                                          params['drug_col_name']: [i.split('|')[1] for i in valSample],
                                          params['res_col_name']: valLabel,
                                          'Prediction': valPredResult}, index=valSample)
        predResult['val'].to_csv(params['output_dir'] + '/' + 'Val_Prediction_Result.txt', header=True,
                                 index=False, sep='\t', line_terminator='\r\n')

    backend.clear_session()

    return history



def main():
    params = initialize_parameters()
    run(params)



if __name__ == "__main__":
    main()
    if K.backend() == "tensorflow":
        K.clear_session()



