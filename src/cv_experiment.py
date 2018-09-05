import os
from os import listdir
from os.path import isfile, join

import numpy as np
import matplotlib.pyplot as plt
import madmom

import sys
sys.path.append('../src')
from preprocessing import RhythmData, SpectroData, MIRData
from models import OLSPatchRegressor, get_model, reset_weights
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
C_values = np.linspace(1., 100, 10)
gamma_values = np.logspace(-10, 0, 10)

for svm_model in svm_models:
    for c in C_values:
        for gamma in gamma_values:
            model_names.append('{}_{}_{}'.format(svm_model, c, gamma))

model_names = ["linear-linvar", "simple_cnn"]

"""
TODO:
- a way to save results, maybe in pandas
- a way to get load or (train and save) a model for the evaluation stuff
"""

def cv_experiment(data, model_name, col_test_data):
   
    epochs=20
    batch_size=8

    input_shape = data.X.shape[1:]
    model = get_model(model_name, input_shape)

    def get_fresh_model(model=model):
        reset_weights(model)
        return model

    train_model = lambda model, X, Y: model.fit(X, Y,
                                        batch_size=batch_size,
                                        epochs=epochs, verbose=0)

    cvacc = cv(data.X, data.Y, get_fresh_model, train_model, nfolds=5, nrepetitions=1)

    return cvacc

    # do cv on the model
    # do cv on TimestepAggregator model
    pass


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

        results[data_name]={}

        if data_name == "columbia-test": continue
        if data_name == "columbia-train": continue
        for Preprocessor in[MIRData]: #  [RhythmData, SpectroData , MIRData]:
            data = Preprocessor(**kwargs)

            col_test_data = Preprocessor(**data_path["columbia-test"])
            for model_name in model_names:
                results['data_name'] = {}
                print("---------------- Experiment for {} on {}({})".format(
                    model_name, Preprocessor.__name__, data_name))
                result = experiment(data, model_name, col_test_data)
                results[data_name][model_name] = result
                print(result)

    return results


if __name__ == "__main__":
    results = run_on_all(cv_experiment)
 
    print('RESULTS:')
    for data_name, data_results in results.items():
        print('-----')
        print(data_name)
        print('-----')

        topacc=0.
        topmod=None
        for model_name, res in data_results.items():
            print('{}: {}'.format(model_name, res))

            if isinstance(res, list) or isinstance(res, np.ndarray):
                acc = res[1]
            else:
                acc = res

            if acc > topacc:
                topacc=acc
                topmod=model_name

        print('-----------------------')
        print('-----------------------')
        print('Finished {}'.format(data_name))
        print('Best model: {}'.format(topmod))
        print('CVAccuracy: {}'.format(topacc))
