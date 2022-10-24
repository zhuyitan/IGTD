from keras import optimizers
from keras.models import Model
from keras.layers import Input, Dense, Dropout, concatenate, Conv2D, BatchNormalization, ReLU, MaxPooling2D, \
    Flatten, AlphaDropout
import numpy as np



def get_DNN_optimizer(params):
    if params['optimizer'] == 'SGD':
        if 'learning_rate' not in params.keys():
            optimizer = optimizers.SGD()
        else:
            optimizer = optimizers.SGD(lr=params['learning_rate'])
    elif params['optimizer'] == 'SGD_momentum':
        if 'learning_rate' not in params.keys():
            optimizer = optimizers.SGD(momentum=0.9)
        else:
            optimizer = optimizers.SGD(lr=params['learning_rate'], momentum=0.9)
    elif params['optimizer'] == 'SGD_momentum_nesterov':
        if 'learning_rate' not in params.keys():
            optimizer = optimizers.SGD(momentum=0.9, nesterov=True)
        else:
            optimizer = optimizers.SGD(lr=params['learning_rate'], momentum=0.9, nesterov=True)
    elif params['optimizer'] == 'RMSprop':
        if 'learning_rate' not in params.keys():
            optimizer = optimizers.RMSprop()
        else:
            optimizer = optimizers.RMSprop(lr=params['learning_rate'])
    elif params['optimizer'] == 'Adagrad':
        if 'learning_rate' not in params.keys():
            optimizer = optimizers.Adagrad()
        else:
            optimizer = optimizers.Adagrad(lr=params['learning_rate'])
    elif params['optimizer'] == 'Adadelta':
        if 'learning_rate' not in params.keys():
            optimizer = optimizers.Adadelta()
        else:
            optimizer = optimizers.Adadelta(lr=params['learning_rate'])
    elif params['optimizer'] == 'Adam':
        if 'learning_rate' not in params.keys():
            optimizer = optimizers.Adam()
        else:
            optimizer = optimizers.Adam(lr=params['learning_rate'])
    elif params['optimizer'] == 'Adam_amsgrad':
        if 'learning_rate' not in params.keys():
            optimizer = optimizers.Adam(amsgrad=True)
        else:
            optimizer = optimizers.Adam(lr=params['learning_rate'], amsgrad=True)
    else:
        if 'learning_rate' not in params.keys():
            optimizer = optimizers.Adam()
        else:
            optimizer = optimizers.Adam(lr=params['learning_rate'])
    return optimizer



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

        self.params['kernel_size'] = []
        self.params['num_kernel'] = []
        self.params['strides'] = []
        for i in range(len(self.params['conv'])):
            kernel_size = []
            strides = []
            num_dim = int((len(self.params['conv'][i][0]) - 1) / 2)
            for j in range(num_dim):
                kernel_size.append(self.params['conv'][i][0][j + 1])
                strides.append(self.params['conv'][i][0][j + 1 + num_dim])
            self.params['kernel_size'].append(kernel_size)
            self.params['strides'].append(strides)
            num_kernel = []
            for j in range(len(self.params['conv'][i])):
                num_kernel.append(self.params['conv'][i][j][0])
            self.params['num_kernel'].append(num_kernel)

        num_kernel_size = len(self.params['kernel_size'])
        num_conv_layer = []
        for i in range(num_kernel_size):
            num_conv_layer.append(len(self.params['num_kernel'][i]))
        num_dense_layer = len(self.params['dense'])

        input = []
        input2List = []
        num_input = len(self.input_data_dim)
        for input_id in range(num_input):
            in_id = Input(shape=(self.input_data_dim[input_id][0], self.input_data_dim[input_id][1], 1),
                          name='Input_' + str(input_id))
            input.append(in_id)
            for j in range(num_kernel_size):
                min_row_size = self.params['pool'][j][0] * 2 + self.params['kernel_size'][j][0] - 1
                min_col_size = self.params['pool'][j][1] * 2 + self.params['kernel_size'][j][1] - 1
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
                    if self.params['cnn_activation'] == 'relu':
                        d = ReLU(name='ReLU_' + str(i) + '_Kernel_' + str(j) + '_Input_' + str(input_id))(d)
                    else:
                        raise TypeError("Activation is not ReLU in subnetwork.")
                    d = MaxPooling2D(pool_size=self.params['pool'][j], name='MaxPooling_' + str(i) + '_Kernel_'
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
                d = Dense(self.params['dense'][i], activation=self.params['activation'], name='Dense_' + str(i),
                          kernel_initializer='lecun_normal')(d)
            else:
                d = Dense(self.params['dense'][i], activation=self.params['activation'], name='Dense_' + str(i))(d)
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
        model.compile(optimizer=get_DNN_optimizer(self.params), loss=self.params['loss'])
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

        self.params['kernel_size'] = []
        self.params['num_kernel'] = []
        self.params['strides'] = []
        for i in range(len(self.params['conv'])):
            kernel_size = []
            strides = []
            num_dim = int((len(self.params['conv'][i][0]) - 1) / 2)
            for j in range(num_dim):
                kernel_size.append(self.params['conv'][i][0][j + 1])
                strides.append(self.params['conv'][i][0][j + 1 + num_dim])
            self.params['kernel_size'].append(kernel_size)
            self.params['strides'].append(strides)
            num_kernel = []
            for j in range(len(self.params['conv'][i])):
                num_kernel.append(self.params['conv'][i][j][0])
            self.params['num_kernel'].append(num_kernel)

        num_kernel_size = len(self.params['kernel_size'])
        num_conv_layer = []
        for i in range(num_kernel_size):
            num_conv_layer.append(len(self.params['num_kernel'][i]))
        num_dense_layer = len(self.params['dense'])

        input = []
        input2List = []
        num_input = len(self.input_data_dim)
        for input_id in range(num_input):
            in_id = Input(shape=(self.input_data_dim[input_id][0], self.input_data_dim[input_id][1], 1),
                          name='Input_' + str(input_id))
            input.append(in_id)
            for j in range(num_kernel_size):
                min_row_size = self.params['pool'][j][0] * 2 + self.params['kernel_size'][j][0] - 1
                min_col_size = self.params['pool'][j][1] * 2 + self.params['kernel_size'][j][1] - 1
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
                    if self.params['cnn_activation'] == 'relu':
                        d = ReLU(name='ReLU_' + str(i) + '_Kernel_' + str(j) + '_Input_' + str(input_id))(d)
                    else:
                        raise TypeError("Activation is not ReLU in subnetwork.")
                    d = MaxPooling2D(pool_size=self.params['pool'][j], name='MaxPooling_' + str(i) + '_Kernel_'
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
                d = Dense(self.params['dense'][i], activation=self.params['activation'], name='Dense_' + str(i),
                          kernel_initializer='lecun_normal')(d)
            else:
                d = Dense(self.params['dense'][i], activation=self.params['activation'], name='Dense_' + str(i))(d)
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
        model.compile(optimizer=get_DNN_optimizer(self.params), loss=self.params['loss'])
        print(model.summary())
        self.model = model
