import os
from os import listdir
from os.path import isfile, join

import numpy as np
import matplotlib.pyplot as plt
import madmom

import sys
sys.path.append('../src')
from preprocessing import RhythmData, SpectroData, MIRData
from models import OLSPatchRegressor, get_model, reset_weights, reshape_keras_conv_input
from utils import cv
import visualize

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import Model
from sklearn.metrics import log_loss

MUSIC = 1
SPEECH = 0

data_path = {
                "GTZAN": {
                    'music_dir': '../data/music_speech/music_wav/',
                    'speech_dir': '../data/music_speech/speech_wav/'
                },
                "columbia-train": {
                    'music_dir': '../data/columbia_music_speech_corpus/train/music',
                    'speech_dir': '../data/columbia_music_speech_corpus/train/speech'
                },
                "columbia-test": {
                    'music_dir': '../data/columbia_music_speech_corpus/test/music/vocals',
                    'speech_dir': '../data/columbia_music_speech_corpus/test/speech'
                }
            }

na = np.newaxis


model_names =[]#  ["meansvm-4.-0.001", "meansvm-10.-0.001", "meansvm"] #  ["linear-linvar", "linear", "simple_cnn"]

# SVM gridsearch values
svm_models = ['meansvm']
C_values     = np.linspace(1., 100, 20)
gamma_values = np.logspace(-10, 0,20)

for svm_model in svm_models:
    for c in C_values:
        for gamma in gamma_values:
            model_names.append('{}_{}_{}'.format(svm_model, c, gamma))

model_names = ["simple_cnn"]

"""
TODO:
- a way to save results, maybe in pandas
- a way to get load or (train and save) a model for the evaluation stuff
"""

def cv_experiment(data, model_name, col_test_data, epochs=20, batch_size=8):

    input_shape = data.X.shape[1:]
    model = get_model(model_name, input_shape)

    def get_fresh_model(model=model):
        reset_weights(model)
        return model

    train_model = lambda model, X, Y: model.fit(X, Y,
                                        batch_size=batch_size,
                                        epochs=epochs, verbose=0)

    cvacc = cv(data.X, data.Y, get_fresh_model, train_model, nfolds=5, nrepetitions=1)

    test_acc = None
    if col_test_data is not None: 
        train_model(model, data.X, data.Y)
        test_acc = evaluate_on_test_set(model, model_name, col_test_data)

    K.clear_session()
    if isinstance(cvacc, np.ndarray)or isinstance(cvacc, list):
        result = (test_acc[1], cvacc[1])
    else:
        result = test_acc, cvacc
    if test_acc is None:
        return result[1]
    else:
        return result

def train_test_experiment(data, model_name, col_test_data, epochs=100, batch_size=8):

    input_shape = data.X.shape[1:]
    model = get_model(model_name, input_shape)

    def get_fresh_model(model=model):
        reset_weights(model)
        return model

    train_model = lambda model, X, Y: model.fit(X, Y,
                                        batch_size=batch_size,
                                        epochs=epochs, verbose=0)

    train_model(model, data.X, data.Y)
    train_acc = model.evaluate(data.X, data.Y)
    
    test_acc = evaluate_on_test_set(model, model_name, col_test_data)
        
    if isinstance(train_acc, np.ndarray)or isinstance(train_acc, list):
        result = (test_acc[1],train_acc[1])
    else:
        result = test_acc, train_acc

    K.clear_session()

    return result

def evaluate_on_test_set(model, model_name, col_test_data):
    
    try:
        test_acc  = model.evaluate(col_test_data.X, col_test_data.Y)
    except ValueError:
        # if the test set has a different time length then the train set, try to reshape the model and test then
        test_input_shape = col_test_data.X.shape[1:]
        model_weights = model.get_weights()
        reshaped_model = reshape_keras_conv_input(model_name, test_input_shape, model_weights)
        test_acc  = reshaped_model.evaluate(col_test_data.X, col_test_data.Y)

    return test_acc

"""
The following stuff happens for trained models
"""
def important_channels(data, model_name, col_test_data):
    pass

def visualize_prediction_over_time(data, model_name, col_test_data):
    # plot for some samples the prediction over time, and also for a transition of music to speech
    # including correlation
    pass

def analyze_error(data, model_name, col_test_data):
    # check false positives/false negatives
    # plot the expected probability of false positives/false negatives under the iid assumption for different timesteps
    pass

def visualize_filter(data, model_name, col_test_data):
    if data.__class__ != SpectroData: return
    pass


def run_on_all(experiment):
    results = {}
    for data_name, kwargs in data_path.items():

        if data_name == "columbia-test": continue # don't use the test set for training
        results[data_name]={}
        for Preprocessor in [MIRData]:# [RhythmData, SpectroData , MIRData]:
            prepr_name = Preprocessor.__name__
            data = Preprocessor(**kwargs)
            
            results[data_name][prepr_name]={}

            col_test_data = Preprocessor(**data_path["columbia-test"])
            for model_name in model_names:
                print("---------------- Experiment for {} on {}({})".format(
                    model_name, Preprocessor.__name__, data_name))
                result = experiment(data, model_name, col_test_data)
                results[data_name][prepr_name][model_name] = result
                print(result)

    return results


if __name__ == "__main__":
    results = run_on_all(train_test_experiment)
    print('\n -------- ') 
    print('|RESULTS:|')
    print(' -------- ') 
    for data_name, data_results in results.items():
        print('\n---> {} <---'.format(data_name))

        topacc=0.
        topmod=None
        tuple_res=False

        for prepr_name, prepr_res in data_results.items():

            for model_name, res in prepr_res.items():

                tuple_res = False
                if isinstance(res, list) or isinstance(res, np.ndarray) or isinstance(res, tuple):
                    acc = res[1]
                    tuple_res = True
                else:
                    acc = res

                if acc > topacc:
                    topacc=acc
                    topres=res
                    topmod=model_name
                    topprepr=prepr_name

        print('Best model: {}'.format(topmod))
        print('(On {})'.format(topprepr))
        print('Model Selection Accuracy: {}'.format(topacc))
        if tuple_res:
            print('Out of sample acc on Columbia-Test: {}'.format(topres[0]))
