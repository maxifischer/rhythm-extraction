import numpy as np
na = np.newaxis

from preprocessing import patch_augment

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import Model

class OLSPatchRegressor():
    def __init__(self):
        self.w = None
        self.filter_shape = None
        
    def fit(self, X, Y, patch_width=10):

        X, Y = patch_augment(X, Y, patch_width)

        self.filter_shape = X.shape[1:]
        X_flattened = X.reshape(X.shape[0],-1)
        
        self.w = (np.linalg.solve(np.dot(X_flattened.T, X_flattened), np.dot(X_flattened.T, Y))).reshape(self.filter_shape)[na,:]

    def predict(self, X, patch_mode=False):
        
        if patch_mode:
            X_flattened = X.reshape(X.shape[0],-1)
            return np.dot(X_flattened, self.w.flatten())
        
        else:
            N, H, W, D = X.shape
            h, w, d = self.filter_shape
            out_shape = (N,W-w+1) 
            y = np.zeros(out_shape)
            
            for i in range(X.shape[1]):
                y[:,i] = np.squeeze(np.tensordot(X[:,:,i:i+w,:], self.w, axes=[[1, 2, 3], [1,2,3]]))
            
            return y 

# KERAS MODELS
def get_keras_model(modelname, input_shape):

    num_frequencies = input_shape[0]

    if modelname == 'simple_cnn':

        time_filter_size=3

        # simple cnn with very narrow filters in the time axis
        model = Sequential()
        model.add(MaxPooling2D(pool_size=(1, 3), input_shape=input_shape))
        model.add(Conv2D(32, kernel_size=(num_frequencies, time_filter_size),
                         activation='relu'))

        model.add(Conv2D(1, kernel_size=(1, 1), activation='sigmoid'))
        model.add(Lambda(lambda x: K.mean(x, axis=[1,2])))

        model.compile(loss=keras.losses.binary_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        
        return model

    elif modelname == 'linear':
        # linear model

        time_filter_size = 3

        model = Sequential()
        model.add(Conv2D(1, kernel_size=(num_frequencies, time_filter_size),
                         activation='sigmoid',
                         input_shape=input_shape))

        model.add(Lambda(lambda x: K.mean(x, axis=[1,2])))

        model.compile(loss=keras.losses.binary_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        
        return model

    else:
        print('modelname unknown')
        return None

def reset_weights(model):
    session = K.get_session()
    for layer in model.layers: 
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)
