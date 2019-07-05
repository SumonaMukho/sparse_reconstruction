import numpy as np
from numpy import genfromtxt
from keras.layers import Input, Dense
from keras.models import Model
import scipy.io as sio


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
print('np.shape(y_train)', np.shape(y_train))
print('np.shape(x_train)', np.shape(x_train))





encoding_dim = 4  # this is our input placeholder
input_img = Input(shape=(nfeatures,))
# "encoded" is the encoded representation of the input
encoded = Dense(36, activation='tanh')(input_img)
encoded=Dense(18,activation='tanh')(encoded)
encoded=Dense(9,activation='tanh')(encoded)
encoded=Dense(encoding_dim,activation='tanh')(encoded)
# "decoded" is the lossy reconstruction of the input
encoded=Dense(9,activation='tanh')(encoded)
decoded = Dense(18, activation='tanh')(encoded)
decoded = Dense(36, activation='tanh')(decoded)
decoded = Dense(nfeatures, activation=None)(decoded)


# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

encoder = Model(input_img, encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))

# retrieve the last layer of the autoencoder model
autoencoder.compile(optimizer='adadelta', loss='mse')

autoencoder.fit(x_train, y_train,
                epochs=10,
                batch_size=100,
                shuffle=True,
                validation_data=(x_train, y_train))





