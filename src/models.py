import numpy as np
na = np.newaxis

from preprocessing import patch_augment

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import Model

from sklearn.metrics import log_loss
from sklearn.svm import SVC

class OLSPatchRegressor():
    def __init__(self, patch_width=5):
        self.w = None
        self.filter_shape = None
        self.patch_width=patch_width
        
    def fit(self, X, Y, **kwargs):
        patch_width=self.patch_width

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

    def evaluate(self, X, y, **kwargs):
        y_pred = np.sign(self.predict(X))
        return np.mean(y_pred == y)

class PatchSVM():
    def __init__(self, C=1., patch_width=5):
        self.C=C
        self.svm=SVC()
        self.patch_width=patch_width

    def fit(self, X, Y, **kwargs):
        patch_width=self.patch_width
        X_patched, Y_patched = patch_augment(X, Y, patch_width)
        X_patched = X_patched.reshape(X_patched.shape[0], -1) 

        self.svm.fit(X_patched, Y_patched)

    def predict(self, X):
            N, H, W, D = X.shape
            out_shape = (N,W-self.patch_width+1) 
            y = np.zeros(out_shape)
            
            for i in range(X.shape[1]):
                y[:,i] = np.squeeze( self.svm.predict(X[:,:,i:i+w,:].reshape[N,-1]))

            return np.sign(np.mean(y, axis=1))

    def evaluate(self, X, Y, **kwargs):
        y_pred = self.predict(X)
        return np.mean(y_pred == Y)

def get_model(modelname, input_shape):

    num_frequencies = input_shape[0]

    if modelname == 'patchsvm':
        return PatchSVM()

    elif modelname == 'patchregressor':
        return OLSPatchRegressor()

    elif modelname == 'simple_cnn':

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

    elif modelname.endswith('-linvar'):
        return TimestampAggregator(lambda: get_model(modelname.split('-linvar')[0], input_shape))
    
    else:
        print('modelname unknown')
        return None

def reset_weights(model):

    if isinstance(model, PatchSVM):
        return PatchSVM(C=model.C, patch_width=model.patch_width)

    elif isinstance(model, OLSPatchRegressor):
        return OLSPatchRegressor(patch_width=model.patch_width)


    elif isinstance(model, TimestampAggregator):
        model.model = reset_weights(model.model)
        return model

    else:
        session = K.get_session()
        for layer in model.layers: 
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel.initializer.run(session=session)

        return model


class TimestampAggregator():
    def __init__(self, get_model):
        self.model = get_model()
        self.time_model = Model(inputs=self.model.input,
                  outputs=self.model.layers[-2].output)
    
    def fit(self, X, Y, *args, **kwargs):
        self.model.fit(X, Y, *args, **kwargs)
        p = self.time_model.predict(X)[:,0,:,0] # (samples, timestamps)
        mean = np.mean(p, axis=1)
        var = np.var(p, axis=1)
        a = np.array([mean, var]).T # (samples, 2)
        # a*w = y = w = (a^t a)^-1 a^t y
        self.w = np.linalg.lstsq(a, Y)[0]
        # for analysis purpose
        self.avg_mean_pos = np.mean(mean[Y>0])
        self.avg_mean_neg = np.mean(mean[Y<=0])
        self.avg_var_pos = np.mean(var[Y>0])
        self.avg_var_neg = np.mean(var[Y<=0])
        
    def predict(self, X):
        p = self.time_model.predict(X)[:,0,:,0] # (samples, timestamps)
        mean = np.mean(p, axis=1) 
        var = np.var(p, axis=1)
        a = np.array([mean, var]).T # (samples, 2)
        return a.dot(self.w)
    
    def evaluate(self, X, Y, *args, **kwargs):
        Y_ = self.predict(X)
        return log_loss(Y, Y_), np.mean((Y_>0.5)==(Y>0.5))
