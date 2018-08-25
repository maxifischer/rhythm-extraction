import os
from os import listdir
from os.path import isfile, join

import numpy as np
import matplotlib.pyplot as plt
import madmom

import sys
sys.path.append('../src')
from preprocessing import RhytmData, SpectroData, MIRData
from models import OLSPatchRegressor
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
                    'music': '../data/music_speech/music_wav/',
                    'speech': '../data/music_speech/speech_wav/'
                },
                "columbia-train": {
                    'music': '../data/columbia_speech_music_corpus/train/music',
                    'speech': '../data/columbia_speech_music_corpus/train/speech'
                },
                "columbia-test": {
                    'music': '../data/columbia_speech_music_corpus/test/music',
                    'speech': '../data/columbia_speech_music_corpus/test/speech'
                }
            }

na = np.newaxis

model_names = ["linear", "simple_cnn"]

def cv_experiment(data, model_name, col_test_data):
    # do cv on the model

    # do cv on TimestepAggregator model

def run_on_all(experiment):
    results = {}
    for data_name, kwargs in data_path.items():
        if data_name == "columbia-test": continue
        for Preprocessor in [RhytmData, SpectroData, MIRData]:
            data = Preprocessor(**kwargs)
            col_test_data = Preprocessing(**data_path["columbia-test"])
            for model_name in model_names:
                print("---------------- Experiment for {} on {}({})".format(
                    model_name, Preprocessor.__class__.__name__, data_name))
                result = experiment(data, model_name, col_test_data)



