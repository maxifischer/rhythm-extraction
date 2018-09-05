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
    def __init__(self, patch_width=5, k=0.001):
        self.w = None
        self.filter_shape = None
        self.patch_width=patch_width
        self.k=k
        
    def fit(self, X, Y, **kwargs):
        patch_width=self.patch_width

        X, Y = patch_augment(X, Y, patch_width)

        self.filter_shape = X.shape[1:]
        X_flattened = X.reshape(X.shape[0],-1)

        N = X_flattened.shape[1]
        
        self.w = (np.linalg.solve(np.dot(X_flattened.T, X_flattened) + self.k * np.eye(N), np.dot(X_flattened.T, Y))).reshape(self.filter_shape)[na,:]

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
        y_pred = (np.sign(self.predict(X)-.5)+ 1)/2
        return np.mean(y_pred == y)

class PatchSVM():
    def __init__(self, C=10, patch_width=100, patch_stride=100, kernel='rbf', gamma=1e-5):
        self._kwargs = {'C':C, 'patch_width':patch_width, 'patch_stride':patch_stride, 'kernel':kernel, 'gamma':gamma}
        self.C=C
        self.svm=SVC(C=C, kernel=kernel, degree=1, gamma=gamma)
        self.patch_width=patch_width
        self.patch_stride=patch_stride

    def fit(self, X, Y, **kwargs):

        #print('labels pre:')
        #print(np.min(Y), np.max(Y))
        ## convert labels to [-1, 1] range
        Y = (Y - .5) * 2
        print('labels post:')
        print(np.min(Y), np.max(Y))

        patch_width=self.patch_width
        X_patched, Y_patched = patch_augment(X, Y, patch_width, patch_stride=self.patch_stride)
        print('...finished patchaugment')
        X_patched = X_patched.reshape(X_patched.shape[0], -1) 
        print('x_patched shape is: {}'.format(X_patched.shape))
        
        self.svm.fit(X_patched, Y_patched)
        print('...finished fitting')
        print('train accuracy: {}'.format(self.svm.score(X_patched, Y_patched)))

    def predict(self, X, prediction_stride=-1):
            N, H, W, D = X.shape
            if prediction_stride < 0:
                prediction_stride = self.patch_width
            out_shape = (N,(W-self.patch_width)//prediction_stride+1) 
            y = np.zeros(out_shape)
            
            for i in range(out_shape[1]):
                strt = i * prediction_stride
                y[:,i] = self.svm.predict(X[:,:,strt:strt+self.patch_width,:].reshape(N,-1))
    
            print('prediction range: [{},{}]'.format(np.min(y), np.max(y)))
            mean_pred = np.mean(y, axis=1)
            print('meaned prediction range: [{},{}]'.format(np.min(mean_pred), np.max(mean_pred)))

            return (np.sign(mean_pred-.5)+1)/2# (np.sign(np.mean(y, axis=1)) + 1) / 2

    def evaluate(self, X, Y, **kwargs):
        y_pred = self.predict(X)
        return np.mean(y_pred == Y)

class MeanSVM():
    def __init__(self, C=10, kernel='rbf', gamma=0.000001):

        print('initialize meansvm with C={}, kernel={}, gamma={}'.format(C, kernel, gamma))

        self._kwargs = {'C':C, 'kernel':kernel, 'gamma':gamma}
        self.C=C
        self.svm=SVC(C=C, kernel=kernel, degree=1, gamma=gamma)

    def mean_time_axis(self, X):
        N,H,W,D = X.shape
        # mean input data over the time axis (W)
        X_meaned = np.mean(X, axis=2).reshape(N, -1) 
        return X_meaned

    def fit(self, X, Y, **kwargs):
        print('input shape: {}'.format(X.shape))

        X_meaned = self.mean_time_axis(X) 
        print('meaned shape: {}'.format(X_meaned.shape))
        self.svm.fit(X_meaned, Y)
        # print train error
        print('...finished fitting')
        print('...training acc: {}'.format(self.svm.score(X_meaned, Y)))

    def predict(self, X, **kwargs):
        X_meaned = self.mean_time_axis(X)
        return self.svm.predict(X_meaned)
    def evaluate(self, X, Y, **kwargs):
        y_pred = self.predict(X)
        return np.mean(y_pred == Y)

def get_model(modelname, input_shape):

    num_frequencies = input_shape[0]


    if modelname.startswith('meansvm') or modelname.startswith('patchsvm'):
        
        if modelname == 'patchsvm':
            return PatchSVM()
        elif  modelname == 'meansvm':
            return MeanSVM()
        else:
            mdl_cnfg = modelname.split('_')
            mdlnme = mdl_cnfg[0]
            c     = float(mdl_cnfg[1])
            gamma = float(mdl_cnfg[2])

            if mdlnme == 'meansvm':
                return MeanSVM(C=c, kernel='rbf', gamma=gamma)
            elif mdlnme == 'patchsvm':
                return PatchSVM(C=c, kernel='rbf', gamma=gamma)
            else:
                print('Modelname unknown: stick to format model-C_VALUE-GAMMA_VALUE ')
                return None

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
        return PatchSVM(**model._kwargs)
    elif isinstance(model, MeanSVM):
        return MeanSVM(**model._kwargs)

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

