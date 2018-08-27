import os
from os import listdir
from os.path import isfile, join

import numpy as np
import matplotlib.pyplot as plt
import madmom

import sys
sys.path.append('../src')
from preprocessing import RhythmData, SpectroData, MIRData
from models import OLSPatchRegressor, get_keras_model, reset_weights
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

model_names = ["linear", "simple_cnn"]

def cv_experiment(data, model_name, col_test_data):
   
    epochs=20
    batch_size=16

    input_shape = data.X.shape[1:]
    model = get_keras_model(model_name, input_shape)

    def get_model(model=model):
        reset_weights(model)
        return model

    train_model = lambda model, X, Y: model.fit(X, Y,
                                        batch_size=batch_size,
                                        epochs=epochs,
                                        verbose=0)

    cvacc = cv(data.X, data.Y, get_model, train_model, nfolds=5, nrepetitions=1)

    return cvacc

    # do cv on the model

    # do cv on TimestepAggregator model

def run_on_all(experiment):
    results = {}
    for data_name, kwargs in data_path.items():

        if data_name == "columbia-test": continue
        if data_name == "columbia-train": continue
        for Preprocessor in [RhythmData, SpectroData]:# , MIRData]:
            data = Preprocessor(**kwargs)

            print(os.getcwd())

            col_test_data = Preprocessor(**data_path["columbia-test"])
            for model_name in model_names:
                print("---------------- Experiment for {} on {}({})".format(
                    model_name, Preprocessor.__class__.__name__, data_name))
                result = experiment(data, model_name, col_test_data)
                results.append(result)
                print(result)

    return results

#  def run_on_one(experiment, data_name):


    

if __name__ == "__main__":
    run_on_all(cv_experiment)
