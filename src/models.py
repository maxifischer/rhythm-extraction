import numpy as np
na = np.newaxis
import tensorflow as tf

from preprocessing import patch_augment

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


class ConvNet():
    def __init__(self, X, y):
