import os
from os import listdir
from os.path import isfile, join

import numpy as np
import matplotlib.pyplot as plt
import madmom

import sys
sys.path.append('../src')
from preprocessing import RhythmData, SpectroData, MIRData
from models import *
from utils import cv
import visualize

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import Model
from sklearn.metrics import log_loss, confusion_matrix

import matplotlib

import pickle
import pandas as pd
pd.set_option('display.max_columns', 15)

import pdb
import random

MUSIC = 1
SPEECH = 0

RUN_NAME='classaccs_test'

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


model_names = []#["meansvm--4.--0.001", "meansvm--10.--0.001", "meansvm", "linear--linvar", "linear", "simple_cnn", "simple_cnn--linvar"]

# SVM gridsearch values
svm_models = ['meansvm']# 'patchsvm']
C_values     = np.linspace(1, 100, 20)# [96.]  #np.linspace(1., 100, 2)
gamma_values = np.logspace(-10, 0, 30)# [1e-6] #np.logspace(-10, 0, 2)

for svm_model in svm_models:
    for c in C_values:
        for gamma in gamma_values:
            model_names.append('{}--{}--{}'.format(svm_model, c, gamma))

# model_names.extend(["simple_cnn--linvar", "simple_cnn", "linear--linvar"])# "linear", "linear--linvar", "simple_cnn", "simple_cnn--linvar"]

model_names = ["simple_cnn"]# , "simple_cnn--linvar"]

def cv_experiment(data, model_name, col_test_data, epochs=100, batch_size=8, nfolds=2, nrepetitions=1):
    """

    :return: ((test accuracy on columbia test data set, f1 score on col test, test acc on colubia test positive class only, test acc on columbia test negative class only), (cv acc on train data, cv f1 score on train data, cv acc on train data pos class only, cv acc on train data neg class only))
    """

    input_shape = data.X.shape[1:]
    model = get_model(model_name, input_shape)

    def get_fresh_model(model=model):
        reset_weights(model)
        return model

    train_model = lambda model, X, Y: model.fit(X, Y,
                                        batch_size=batch_size,
                                        epochs=epochs, verbose=0)

    cvacc = cv(data.X, data.Y, get_fresh_model, train_model, nfolds=nfolds, nrepetitions=nrepetitions)

    test_acc = None
    if col_test_data is not None: 
        train_model(model, data.X, data.Y)

        test_acc = evaluate_on_test_set(model, model_name, col_test_data)

    print('cvacc output is of length {}'.format(len(cvacc)))  

    K.clear_session()
    if len(cvacc) == 5:
        result = (test_acc[1:], cvacc[1:])
    else:
        result = test_acc, cvacc
    return result


def train_test_experiment(data, model_name, col_test_data, epochs=100, batch_size=8):

    input_shape = data.X.shape[1:]
    model = get_model(model_name, input_shape)

    def get_fresh_model(model=model):
        reset_weights(model)
        return model

    train_model = lambda model, X, Y: model.fit(X, Y,
                                        batch_size=batch_size,
                                        epochs=epochs)

    train_model(model, data.X, data.Y)
    train_acc = model.evaluate(data.X, data.Y)
    
    test_acc = evaluate_on_test_set(model, model_name, col_test_data)
        
    if len(train_acc) == 5:
        result = (test_acc[1:], train_acc[1:])
    else:
        result = test_acc, train_acc

    K.clear_session()

    return result

def evaluate_on_test_set(model, model_name, col_test_data, return_conf_matrix=False):
    
    try:

        if return_conf_matrix:
            test_predictions  = model.predict(col_test_data.X)
            test_acc = confusion_matrix(col_test_data.Y, test_predictions)
        else:
            test_acc  = model.evaluate(col_test_data.X, col_test_data.Y)
    except ValueError:
        # if the test set has a different time length then the train set, try to reshape the model and test then
        test_input_shape = col_test_data.X.shape[1:]
        model_weights = model.get_weights()
        reshaped_model = reshape_keras_conv_input(model_name, test_input_shape, model_weights)

        if return_conf_matrix:
            test_predictions = reshaped_model.predict(col_test_data.X)
            test_acc = confusion_matrix(col_test_data.Y, test_predictions)
        else:
            test_acc = reshaped_model.evaluate(col_test_data.X, col_test_data.Y)

    return test_acc

"""
The following stuff happens for trained models
"""
def get_trained_model(data, model_name, epochs=100, batch_size=8):
    input_shape = list(data.X.shape[1:])
    model = get_model(model_name, input_shape)
    result_dir = "../results/{}--{}".format(data.__class__.__name__, model_name)
    os.makedirs(result_dir, exist_ok=True)
    weight_path = join(result_dir, "weights.h5")

    def get_fresh_model(model=model):
        reset_weights(model)
        return model

    train_model = lambda model, X, Y: model.fit(X, Y,
                                                batch_size=batch_size,
                                                epochs=epochs, verbose=0)
    try:
        model.load_weights(weight_path)
    except:
        train_model(model, data.X, data.Y)
        try:
            model.save(weight_path)
        except: pass
    return model

def get_accuracy(data, model_name, model, col_test_data):
    r = cv_experiment(data, model_name, col_test_data)
    result_dir = "../results/{}--{}".format(data.__class__.__name__, model_name)
    with open(join(result_dir, "accuracy.txt"), "w") as f:
        f.write("Test Accuracy: {}\nTest f1 Score: {}\nCV Accuracy: {}\nCV f1 score {}".format(
            r[0][0], r[0][1], r[1][0], r[1][1]))


def important_channels(data, model_name, model, col_test_data):
    result_dir = "../results/{}--{}".format(data.__class__.__name__, model_name)
    # TODO
    pass

def visualize_prediction_over_time(data, model_name, model, music_sample, speech_sample):
    result_dir = "../results/{}--{}".format(data.__class__.__name__, model_name)
    # plot for some samples the prediction over time, and also for a transition of music to speech
    # including correlation
    visualize.prediction_over_time(music_sample, speech_sample, model, result_dir)

def analyze_error(data, model_name, model, col_test_data):
    result_dir = "../results/{}--{}".format(data.__class__.__name__, model_name)
    # check false positives/false negatives
    # plot the expected probability of false positives/false negatives under the iid assumption for different timesteps
    # TODO
    pass

def visualize_filter(data, model_name, model, col_test_data, music_sample, speech_sample):
    original_len = music_sample.shape[1]
    target_len = model.input_shape[2]
    T_ORIGINAL = 30*original_len/target_len
    if original_len < target_len:
        m, s = np.zeros(model.input_shape[1:]), np.zeros(model.input_shape[1:])
        m[:, :original_len] = music_sample
        m[:, original_len:] = music_sample
        s[:, :original_len] = speech_sample
        s[:, original_len:] = speech_sample
        music_sample, speech_sample = m, s
    music_sample = music_sample[None, ...]
    speech_sample = speech_sample[None, ...]
    result_dir = "../results/{}--{}".format(data.__class__.__name__, model_name)
    if data.__class__ != SpectroData or isinstance(model, (PatchSVM, MeanSVM)): return

    W_all, i = None, -1
    while W_all is None:
        i += 1
        try:
            W_all = model.layers[i].get_weights()[0]
        except:
            pass

    inspect_model = Model(inputs=model.input, outputs=model.layers[i].output)
    music_activation = inspect_model.predict(music_sample)[0, 0]
    speech_activation = inspect_model.predict(speech_sample)[0, 0]

    time = np.arange(0, T_ORIGINAL, T_ORIGINAL / music_activation.shape[0])

    num_output_channels = W_all.shape[-1]

    for output_channel in range(num_output_channels):
        W = W_all[:, :, :, output_channel]
        bound = np.max(np.absolute(W))
        norm = matplotlib.colors.Normalize(vmin=-bound, vmax=bound)

        num_filters = W.shape[-1]
        num_subplots = 3 * num_filters

        fig = plt.figure(figsize=(12, 9))

        for channel in range(num_filters):
            w_channel = W[:, :, channel]
            w_plus = np.maximum(w_channel, 0)
            w_minus = -np.maximum(-w_channel, 0)
            plt.subplot(num_filters + 1, 3, 1 + channel * 3)
            plt.imshow(w_channel, cmap="PuOr", norm=norm)
            plt.colorbar()

            plt.subplot(num_filters + 1, 3, 2 + channel * 3)
            plt.imshow(w_plus, cmap="PuOr", norm=norm)
            plt.colorbar()

            if model_name == 'linear':
                plt.title("Evidence for music")

            plt.subplot(num_filters + 1, 3, 3 + channel * 3)
            plt.imshow(w_minus, cmap="PuOr", norm=norm)
            plt.colorbar()
            if model_name == 'linear':
                plt.title("Evidence for speech")

        plt.subplot(num_filters + 1, 1, num_filters + 1)
        plt.plot(time, music_activation[:, output_channel], label="Music")
        plt.plot(time, speech_activation[:, output_channel], label="Speech")
        plt.xlabel("Time/s")
        plt.ylabel("Channel activation")
        plt.legend()

        plt.tight_layout()
        plt.savefig("{}/filter-channel-{}.png".format(result_dir, output_channel))


def plot_channel_activation(music, speech, export_path, seconds=15):
    num_channels = speech.shape[-1]
    time = np.arange(0, seconds, seconds / music.shape[1])
    os.makedirs(export_path, exist_ok=True)

    for channel in range(num_channels):
        plt.plot(time, music[0,:, channel], label="Music")
        plt.plot(time, speech[0,:, channel], label="Speech")
        plt.xlabel("Time/s")
        plt.ylabel("Activation")
        plt.title("Channel {}".format(channel))
        plt.legend()
        plt.savefig(export_path+"/channel-{}-sample.png".format(channel))
        plt.clf()

def plot_channel_activations(all_music, all_speech, export_path, seconds=15):
    num_channels = all_speech.shape[-1]
    time = np.arange(0, seconds, seconds / all_music.shape[2])
    os.makedirs(export_path, exist_ok=True)

    for channel in range(num_channels):
        label = "Music"
        for music in all_music:
            plt.plot(time, music[0,:, channel], color="green", label=label)
            label=None # do only one music label, not multiples
        label = "Speech"
        for speech in all_speech:
            plt.plot(time, speech[0,:, channel], color="red", label=label)
            label=None
        plt.xlabel("Time/s")
        plt.ylabel("Activation")
        plt.title("Channel {}".format(channel))
        plt.legend()
        plt.savefig(export_path+"/channel-{}-all-samples.png".format(channel))
        plt.clf()

def plot_channel_activations_mv(all_music, all_speech, export_path, seconds=15):
    num_channels = all_speech.shape[-1]
    time = np.arange(0, seconds, seconds / all_music.shape[2])
    os.makedirs(export_path, exist_ok=True)

    for channel in range(num_channels):
        music_m = np.mean(all_music[:, 0, :, channel], axis=0)
        music_v = np.var(all_music[:, 0, :, channel], axis=0)
        speech_m = np.mean(all_speech[:, 0, :, channel], axis=0)
        speech_v = np.var(all_speech[:, 0, :, channel], axis=0)

        plt.plot(time, music_m, label="Music", color="green")
        plt.plot(time, music_m+music_v, "--", color="green")
        plt.plot(time, music_m-music_v, "--", color="green")
        plt.plot(time, speech_m, label="Speech", color="red")
        plt.plot(time, speech_m+speech_v, "--", color="red")
        plt.plot(time, speech_m-speech_v, "--", color="red")

        plt.xlabel("Time/s")
        plt.ylabel("Activation")
        plt.title("Channel {}".format(channel))
        plt.legend()
        plt.savefig(export_path+"/channel-{}-mv.png".format(channel))
        plt.clf()


def visualize_channel_activation():
    MUSIC = 1
    SPEECH = 0

    for data_name, kwargs in data_path.items():

        if data_name == "columbia-test": continue  # don't use the test set for training
        for Preprocessor in [RhythmData, MIRData]:
            prepr_name = Preprocessor.__name__
            data = Preprocessor(**kwargs)

            col_test_data = Preprocessor(**data_path["columbia-test"])

            music = col_test_data.X[col_test_data.Y == MUSIC]
            speech = col_test_data.X[col_test_data.Y != MUSIC]
            music_sample = random.choice(music)
            speech_sample = random.choice(speech)

            plot_channel_activation(music_sample, speech_sample, "../results/{}".format(prepr_name))
            plot_channel_activations(music, speech, "../results/{}".format(prepr_name))
            plot_channel_activations_mv(music, speech, "../results/{}".format(prepr_name))


def analyze_trained_models():
    MUSIC = 1
    SPEECH = 0

    for data_name, kwargs in data_path.items():

        if data_name == "columbia-test": continue  # don't use the test set for training
        for Preprocessor in [SpectroData]: #[RhythmData, MIRData, SpectroData]:
            prepr_name = Preprocessor.__name__
            data = Preprocessor(**kwargs)

            col_test_data = Preprocessor(**data_path["columbia-test"])

            music_sample    = random.choice(col_test_data.X[col_test_data.Y == MUSIC])
            speech_sample   = random.choice(col_test_data.X[col_test_data.Y != MUSIC])

            print("Col test data \nX: {}, Y: {}".format(col_test_data.X.shape, col_test_data.Y.shape))


            for model_name in model_names:
                print("\n\n-------\nAnalyze {} on {} - {}".format(model_name, data_name, prepr_name))

                print("music sample: ", music_sample.shape)



                model = get_trained_model(data, model_name)
                #get_accuracy(data, model_name, model, col_test_data)
                visualize_prediction_over_time(data, model_name, model, music_sample, speech_sample)
                analyze_error(data, model_name, model, col_test_data)
                important_channels(data, model_name, model, col_test_data)
                visualize_filter(data, model_name, model, col_test_data, music_sample, speech_sample)



def run_on_all(experiment):
    cols = ['data_name','prepr_name','model_name', 'param_c', 'param_gamma', 'param_linvar', 'test_acc', 'test_f1', 'test_acc_pc', 'test_acc_nc', 'cv_acc', 'cv_f1', 'cv_acc_pc', 'cv_acc_nc']
    results = pd.DataFrame(columns=cols)
    for data_name, kwargs in data_path.items():

        if data_name == "columbia-test": continue # don't use the test set for training
        for Preprocessor in [RhythmData]:#[RhythmData, MIRData, SpectroData]:
            prepr_name = Preprocessor.__name__
            data = Preprocessor(**kwargs)

            col_test_data = Preprocessor(**data_path["columbia-test"])
            for model_name in model_names:
                print("---------------- Experiment for {} on {}({})".format(
                    model_name, Preprocessor.__name__, data_name))
                result = experiment(data, model_name, col_test_data)
                split_model = model_name.split('--')
                print('finished cv, result:')
                print(result)
                
                if split_model[-1] == 'linvar':
                    param_linvar = True
                else:
                    param_linvar = None
                if len(split_model) > 2:
                    param_c, param_gamma = split_model[1:3]
                else:
                    param_c, param_gamma = (None, None)

                model_name = split_model[0]
                df_vals = [data_name, prepr_name, model_name, param_c, param_gamma, param_linvar]
                flattened = [val for sublist in result for val in sublist]
                df_vals.extend(flattened[0:8])
                results = results.append(pd.DataFrame(dict(zip(cols, df_vals)), index=[0]), ignore_index=True)

                #test_acc, test_f1 = result[0][0], result[0][1]
                #cv_acc, cv_f1 = result[1][0], result[1][1]

                #df_vals = [data_name, prepr_name, model_name, param_c, param_gamma, param_linvar, test_acc, test_f1, cv_acc, cv_f1]
                #results.loc[results.shape[0]] = df_vals

                #flattened = [val for sublist in result for val in sublist]
                #df_vals.extend(flattened[0:4])
                #results = results.append(pd.DataFrame(dict(zip(cols, df_vals)), index=[0]), ignore_index=True)
    return results


if __name__ == "__main__":
    visualize_channel_activation()
    #analyze_trained_models()
    exit()

    if not os.path.exists('results'):
        os.mkdir('results')
    save_file_name = 'results/{}_results.csv'.format(RUN_NAME)
   

    if True or not os.path.exists(save_file_name):
        print('no save file found... calc it')
        results = run_on_all(cv_experiment)
        results.to_csv(save_file_name, index=False)
        print('...saved results')
    else:
        results = pd.read_csv(save_file_name)
        print('...loaded results from file')

    print('\n -------- ') 
    print('|RESULTS:|')
    print(' -------- ') 

    best_overall_run = results.iloc[results[["cv_acc"]].idxmax()]
    print(best_overall_run)
    best_run_per_dataset = results.groupby("data_name")["cv_acc"].idxmax()# .apply(np.argmax)
    best_run_per_model = results.groupby("model_name")["cv_acc"].idxmax()# .apply(np.argmax)
    print('Best model: {}'.format(best_overall_run["model_name"]))
    print('(On {})'.format(best_overall_run["prepr_name"]))
    print('Model Selection Accuracy: {}'.format(best_overall_run["cv_acc"]))
    print(results)
