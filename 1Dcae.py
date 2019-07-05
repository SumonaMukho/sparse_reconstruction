import numpy as np
from numpy import genfromtxt
import keras.backend as K
from keras.models import Model
import scipy.io as sio
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, BatchNormalization, LSTM, RepeatVector,Flatten,Reshape

model_params = {'tau': 0.25,
                'nstep': 1000,
                'N': 11,
                'd': 22}

res_params = {'radius': 0.5,
              'degree': 3,
              'sigma': 0.5,
              'train_length': 500000,
              'num_inputs': model_params['N'],
              'predict_length': 2000,
              'beta': 0.001
              }
##
spin_off=0
shift_k=0
nfeatures=72
##

## Read Data
u=sio.loadmat('x_y_combined.mat')
v=u.get('data_mat')
data=np.transpose(v)
train = data[shift_k:shift_k+res_params['train_length'],:]
label = data[1+shift_k:1+shift_k+res_params['train_length'],:]
print('np.shape(train)', np.shape(train))
print('np.shape(label)', np.shape(label))

##
trainN=100000
testN=2000
##
y_train = label - train
x_train = train
x_train=np.reshape(x_train,(res_params['train_length'],nfeatures,1))
y_train=np.reshape(y_train,(res_params['train_length'],nfeatures,1))

#print('np.shape(y_train)', np.shape(y_train))
#print('np.shape(x_train)', np.shape(x_train)

input_signal= Input(batch_shape=(None,nfeatures,1))
#input_sig = Input(shape=(nfeatures,1))
x = Conv1D(36,3, activation='relu', padding='valid')(input_signal)
x1 = MaxPooling1D(2)(x)
x2 = Conv1D(18,3, activation='relu', padding='valid')(x1)
x3 = MaxPooling1D(2)(x2)
flat = Flatten()(x3)
encoded = Dense(9,activation = 'relu')(flat)
encoded = Reshape((9,1))(encoded)
print("shape of encoded {}".format(K.int_shape(encoded)))

x2_ = Conv1D(9, 3, activation='relu', padding='valid')(encoded)
x1_ = UpSampling1D(2)(x2_)
x_ = Conv1D(18, 3, activation='relu', padding='valid')(x1_)
upsamp = UpSampling1D(2)(x_)
x3_=Conv1D(36, 3, activation='relu', padding='valid')(x_)
flat = Flatten()(upsamp)
decoded = Dense(nfeatures,activation = None)(flat)
decoded = Reshape((nfeatures,1))(decoded)

print("shape of decoded {}".format(K.int_shape(decoded)))

autoencoder = Model(input_signal, decoded)
autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
autoencoder.fit(x_train, y_train,
                epochs=1,
                batch_size=100,
                shuffle=True,
                validation_data=(x_train, y_train))

