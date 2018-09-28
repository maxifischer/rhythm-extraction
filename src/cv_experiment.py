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
from utils import cv, normalize_channels, LameModelException
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
from itertools import product

pd.set_option('display.max_columns', 15)

import pdb
import random

import json

MUSIC = 1
SPEECH = 0

RUN_NAME='mv_linear-all'
NORMALIZE_CHANNELS=True

# stop cv iterations and do not save the results if cvacc below threshold
LAME_MODEL_THRESHOLD = 0.

if RUN_NAME == 'mv_mir-rhythm':
    NORMALIZE_CHANNELS=True
    Preprocessors = [RhythmData, MIRData]  # , SpectroData]
    def add_models():
        add_mv_best()
        add_deep_nns()
        #add_mv_grid()
    REPETITIONS = 10
elif RUN_NAME.startswith('mv_spectro'):
    NORMALIZE_CHANNELS=True
    Preprocessors = [SpectroData]  # , SpectroData]
    def add_models():
        #add_mv_best()
        add_mv_grid()
        add_deep_nns()
    REPETITIONS = 10

elif RUN_NAME == 'mv_linear-all':
    NORMALIZE_CHANNELS=True
    Preprocessors = [RhythmData, MIRData, SpectroData]
    def add_models():
        model_names.append("mv_linear")
    REPETITIONS=10

# convolution-style models

elif RUN_NAME == 'cs_mir-rhythm':
    NORMALIZE_CHANNELS=True
    Preprocessors = [RhythmData, MIRData]  # , SpectroData]
    def add_models():
        model_names.extend(["linear", "simple_cnn"])
    REPETITIONS = 10

elif RUN_NAME == 'cs_spectro':
    NORMALIZE_CHANNELS=True
    Preprocessors = [SpectroData]
    def add_models():
        model_names.extend(["linear", "simple_cnn"])
    REPETITIONS = 10

elif RUN_NAME == 'cs_cnn-spectro':
    NORMALIZE_CHANNELS=True
    Preprocessors = [SpectroData]
    def add_models():
        model_names.extend(["simple_cnn"])
    REPETITIONS = 10

elif RUN_NAME == 'cs_linear-spectro':
    NORMALIZE_CHANNELS=True
    Preprocessors = [SpectroData]
    def add_models():
        model_names.extend(["linear"])
    REPETITIONS = 10


else:
    class FlagHaterException(Exception):
        pass
    raise FlagHaterException("you dont like flags or what")

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


model_names = []

def add_deep_nns():
    for layers in range(5, 7):
        model_names.append("mv_nn--{}".format(json.dumps({"hidden_neurons": [100]*layers, "dropout": 0})))
        model_names.append("mv_nn--{}".format(json.dumps({"hidden_neurons": [100]*layers, "dropout": 0.25})))

def add_mv_best():
    df = pd.read_csv("../old/mv-models.csv")
    best = df.loc[df.cv_acc == 1.]
    for _, row in best.iterrows():
        model_names.append(row["model_name"]+"--"+row["hyper_params"])

    print(model_names)


def add_mv_grid():

    C_values = np.linspace(1, 100, 20)  # [96.]  #np.linspace(1., 100, 2)
    gamma_values = np.logspace(-10, 0, 30)  # [1e-6] #np.logspace(-10, 0, 2)
    for c, gamma in product(C_values, gamma_values):
        model_names.append("mv_svm--{}".format(json.dumps({"C": c, "gamma": gamma})))

    # hidden_neurons=[100], dropout=0
    for num_neurons, num_layers, dropout in product([50, 100, 200], [1, 2, 3], [0., 0.25, 0.5]):
        model_names.append("mv_nn--{}".format(json.dumps(
            {"hidden_neurons": [num_neurons] * num_layers,
             "dropout": dropout})))





# SVM gridsearch values
def add_svm_grid():
    svm_models = ['meansvm']# 'patchsvm']
    C_values     = np.linspace(1, 100, 20)# [96.]  #np.linspace(1., 100, 2)
    gamma_values = np.logspace(-10, 0, 30)# [1e-6] #np.logspace(-10, 0, 2)

    for svm_model in svm_models:
        for c in C_values:
            for gamma in gamma_values:
                model_names.append('{}--{}--{}'.format(svm_model, c, gamma))
#model_names = []
#model_names.extend(["simple_cnn--linvar", "simple_cnn", "linear--linvar", "linear"])# "linear", "linear--linvar", "simple_cnn", "simple_cnn--linvar"]




def train_test_experiment(data, model_name, col_test_data, epochs=100, batch_size=8):

    print('ATTENTION: unnormalized stuff working here!!!!!!!!!!!!')

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


def accuracy_on_test_set(model, model_name, col_test_data):
    try:

        P = model.predict(col_test_data.X)
        Y = np.reshape(col_test_data.Y, P.shape)
        return np.mean((P>0.5)==(Y>0.5))
    except ValueError:
        # if the test set has a different time length then the train set, try to reshape the model and test then
        test_input_shape = col_test_data.X.shape[1:]
        model_weights = model.get_weights()
        model = reshape_keras_conv_input(model_name, test_input_shape, model_weights)

        P = model.predict(col_test_data.X)
        Y = np.reshape(col_test_data.Y, P.shape)
        return np.mean((P > 0.5) == (Y > 0.5))

def evaluate_on_test_set(model, model_name, Xtest, Ytest, return_conf_matrix=False):
    try:

        P = model.predict(Xtest)
        Y = np.reshape(Ytest, P.shape)

        if return_conf_matrix:
            test_acc = confusion_matrix(Ytest, P)
        else:
            test_acc = model.evaluate(Xtest, Y)
    except ValueError:
        # if the test set has a different time length then the train set, try to reshape the model and test then
        test_input_shape = Xtest.shape[1:]
        model_weights = model.get_weights()
        model = reshape_keras_conv_input(model_name, test_input_shape, model_weights)

        P = model.predict(Xtest)
        Y = np.reshape(Ytest, P.shape)

        if return_conf_matrix:
            test_acc = confusion_matrix(Ytest, P)
        else:
            test_acc = model.evaluate(Xtest, Y)
    if test_acc[0] < 0.5:
        print("this is fucked:", model_name, test_acc[0])
        # pdb.set_trace()
        print("P", P.shape)
        print("Y", Y.shape)


    return test_acc

"""
The following stuff happens for trained models
"""
def get_trained_model(data, model_name, col_test_data=None, retrain=False, epochs=100, batch_size=8):
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

    Xtrain, stddev = normalize_channels(data.X.copy())
    Ytrain = data.Y
    if col_test_data:
        Xtest = col_test_data.X.copy() / stddev
        Ytest = col_test_data.Y
    else:
        Xtest, Ytest=None, None

    try:
        assert not retrain
        model.load_weights(weight_path)
        print("loaded model")
    except:
        print("train model")

        train_model(model, Xtrain, Ytrain)
        try:
            model.save(weight_path)
        except: pass
    return model, Xtrain, Ytrain, Xtest, Ytest

def get_accuracy(data, model_name, model, col_test_data):
    r = cv_experiment(data, model_name, col_test_data)
    result_dir = "../results/{}--{}".format(data.__class__.__name__, model_name)
    with open(join(result_dir, "accuracy.txt"), "w") as f:
        f.write("Test Accuracy: {}\nTest f1 Score: {}\nCV Accuracy: {}\nCV f1 score {}".format(
            r[0][0], r[0][1], r[1][0], r[1][1]))


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

            #plot_channel_activation(music_sample, speech_sample, "../results/{}".format(prepr_name))
            #plot_channel_activations(music, speech, "../results/{}".format(prepr_name))
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







def cv_experiment(data, model_name, col_test_data, epochs=100, batch_size=8, nfolds=5, nrepetitions=REPETITIONS, norm_channels=NORMALIZE_CHANNELS):
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

    cvacc = cv(data.X, data.Y, get_fresh_model, train_model, nfolds=nfolds, nrepetitions=nrepetitions, norm_channels=norm_channels, lame_model_threshold=LAME_MODEL_THRESHOLD)

    test_acc = None
    if col_test_data is not None:
        if norm_channels:
            Xtrain, stddev = normalize_channels(data.X.copy())
            Ytrain = data.Y
            Xtest  = col_test_data.X.copy() / stddev
            Ytest  = col_test_data.Y
        else:
            Xtrain = data.X
            Ytrain = data.Y
            Xtest  = col_test_data.X
            Ytest  = col_test_data.Y

        train_model(model, Xtrain, Ytrain)

        test_acc = evaluate_on_test_set(model, model_name, Xtest, Ytest)

    print('cvacc output is of length {}'.format(len(cvacc)))

    K.clear_session()
    if len(cvacc) == 5:
        result = (test_acc[1:], cvacc[1:])
    else:
        result = test_acc, cvacc
    print(">>>>>>>>", model_name, cvacc[0])
    return result


def run_on_all(experiment):
    cols = ['data_name','prepr_name','model_name', 'param_c', 'param_gamma', 'param_linvar', 'test_acc', 'test_f1',
            'test_acc_pc', 'test_acc_nc', 'cv_acc', 'cv_f1', 'cv_acc_pc', 'cv_acc_nc', 'hyper_params', 'is_normalized']
    results = pd.DataFrame(columns=cols)
    for data_name, kwargs in data_path.items():

        if data_name == "columbia-test": continue # don't use the test set for training
        for Preprocessor in Preprocessors:
            prepr_name = Preprocessor.__name__
            data = Preprocessor(**kwargs)

            col_test_data = Preprocessor(**data_path["columbia-test"])
            for model_id, model_name in enumerate(model_names):
                print("---------------- Experiment for {} on {}({})".format(
                    model_name, Preprocessor.__name__, data_name))
                try:
                    result = experiment(data, model_name, col_test_data)
                except LameModelException as e:
                    print(type(e), e)
                    continue
                split_model = model_name.split('--')
                print('finished cv, result:')
                print(result)

                hyper_params = ""

                if split_model[-1] == 'linvar':
                    param_linvar = True
                else:
                    param_linvar = None

                if split_model[0].startswith("mv"):
                    try:
                        hyper_params = split_model[1]
                    except: pass

                param_c, param_gamma = (None, None)
                if model_name.startswith("mean"):
                    if len(split_model) > 2:
                        param_c, param_gamma = split_model[1:3]

                test_acc, test_f1, test_acc_pc, test_acc_nc = result[0]
                cv_acc, cv_f1, cv_acc_pc, cv_acc_nc = result[1]


                model_name = split_model[0]
                vals = pd.DataFrame({'data_name': [data_name],'prepr_name': [prepr_name],'model_name': [split_model[0]], 'param_c': [param_c],
                       'param_gamma': [param_gamma], 'param_linvar': [param_linvar], 'test_acc': [test_acc],
                       'test_f1': [test_f1], 'test_acc_pc': [test_acc_pc], 'test_acc_nc': [test_acc_nc],
                       'cv_acc': [cv_acc], 'cv_f1': [cv_f1], 'cv_acc_pc': [cv_acc_pc], 'cv_acc_nc': [cv_acc_nc],
                       'hyper_params': [hyper_params], 'is_normalized': [1 if NORMALIZE_CHANNELS else 0]})
                results = results.append(vals, ignore_index=True)


                if model_id % 100 == 0:
                    print(results)
                    results.to_csv("cv_temp_results.csv", index=False)

    return results


if __name__ == "__main__":
    cmd = sys.argv[1]

    open_csv = None
    if cmd == "cv":
        add_models()
        print("CV for ", model_names)
        if not os.path.exists('results'):
            os.mkdir('results')
        save_file_name = 'results/{}.csv'.format(RUN_NAME)

        results = run_on_all(cv_experiment)

        results["is_normalized"] = 1 if NORMALIZE_CHANNELS else 0

        results.to_csv(save_file_name, index=False)

        open_csv = save_file_name

    if cmd == "merge_csv":
        all = []
        for file in os.listdir("results"):
            try:
                add = pd.read_csv(join("results", file))
                if not "hyper_params" in add:
                    add["hyper_params"] = ""
                all.append(add)
            except Exception as e:
                print("Skipped {} bc {} ({})".format(file, e, type(e)))
        merged = pd.concat(all)
        merged.to_csv("results/merged.csv", index=False)
        open_csv = "results/merged.csv"


    if cmd == "open_csv":
        open_csv = sys.argv[2]

    if open_csv:
        save_file_name = open_csv
        results = pd.read_csv(save_file_name)

        results = results.dropna(subset=["cv_acc"])
        results["param_linvar"].fillna(False, inplace=True)


        #results = results.sort_values(["cv_acc", "test_acc"], ascending=False)

        #results = results.drop(
        #    results.loc[results.cv_acc].index
        #)

        models = results["model_name"].unique()
        datasets = results["data_name"].unique()
        preprocessing = results["prepr_name"].unique()

        print("Evaluation of: \n  models: {}\n  datasets: {}\n  prepr: {}\n".format(models, datasets, preprocessing))

        print('...loaded results from file')

        print('\n -------- ')
        print('|RESULTS:|')
        print(' -------- ')

        best_overall_run = results.iloc[results[["cv_acc"]].idxmax()]
        print("highest cv_accuracy:",  best_overall_run)

        print("\n===============================")
        print("Best run per dataset cv_acc:")
        best_run_per_dataset = results.groupby("data_name")["cv_acc"].idxmax()  # .apply(np.argmax)
        for row in best_run_per_dataset:
            v = results.iloc[row]
            print("{}: {} on {} with {} ({}, {})".format(v["data_name"], v["model_name"], v["prepr_name"], v["cv_acc"],
                                                     "normalized" if v["is_normalized"] == 1 else "unnorm.",
                                                     "linvar" if v["param_linvar"] else "mean"   ))
            print(v)
            print("\n")

        print("\n\n===============================")

        print("Best run per dataset test_acc:")
        best_run_per_dataset = results.groupby("data_name")["test_acc"].idxmax()  # .apply(np.argmax)
        for row in best_run_per_dataset:
            v = results.iloc[row]
            print("{}: {} on {} with {} ({}, {})".format(v["data_name"], v["model_name"], v["prepr_name"], v["test_acc"],
                                                     "normalized" if v["is_normalized"] == 1 else "unnorm.",
                                                     "linvar" if v["param_linvar"] else "mean"   ))
            print(v)
            print("\n")

        def best_setup_latex(group_by, metric_name, result_cols):

            def clean_for_latex(word, num_digits=4):
                if isinstance(word, str):
                    # escape _ character for latex
                    return word.replace("_", "\\_")
                if isinstance(word, float):
                    # cut floating points to num_digits digits
                    return "{:.4f}".format(word).rstrip("0")
                else:
                    return word

            result_headers = group_by + result_cols
            result_headers_latex_filtered = [clean_for_latex(v) for v in result_headers]
            print("==========================")
            print("\\begin{{tabular}}{{{}}}".format("|".join(["c" for _ in result_headers])))
            best_runs = results.groupby(group_by)[metric_name].idxmax()
            print("& ".join(result_headers_latex_filtered), r"\\")
            print("\\hline")
            for row in best_runs:
                rw = results.iloc[row]
                print("& ".join([clean_for_latex(rw[v]) for v in result_headers]), r"\\")
            print("\\end{tabular}")
            print("==========================")

        def best_setup(group_by, metric_name, result_cols):
            print("==========================")
            best_runs = results.groupby(group_by)[metric_name].idxmax()
            for idx in best_runs:
                row = results.iloc[idx]
                print(row[group_by])
                print("->")
                print(row[result_cols])
                print("----\n")
            print("==========================")

        print('As latex: (cvacc)')
        best_setup_latex(["data_name"], "cv_acc", ["model_name", "cv_acc", "cv_f1", "prepr_name", "test_acc", "test_f1"])
#         rows = results.loc[(results.cv_acc==1.) & (results.test_acc==1.)]
#         print("Perfect Setups (cv_acc=test_acc=1)", rows[["data_name", "prepr_name", "model_name", "hyper_params"]])
#         """
#         #Best model for each preprocessing type
#         metric = "cv_acc"
#         best_runs = results.groupby(["data_name", "prepr_name"])[metric].idxmax()
#         for idx in best_runs:
#             row = results.iloc[idx]
#             print(row[["data_name", "prepr_name"]])
#             print(row[["model_name", "param_linvar", "cv_acc"]])
#             print("----")
#         """
#         print("****************************")
#         print("****************************")
#         print("Best model per feature set")
#         best_setup(["data_name", "prepr_name"], "cv_acc", ["model_name", "param_linvar", "cv_acc", "test_acc"])

        print('Best run, without linvar option')
        results_no_mv_post = results[results["param_linvar"] == False].groupby("data_name")["cv_acc"].idxmax()
        for row in results_no_mv_post:
            v = results.iloc[row]
            # print("{}: {} on {} with {} ({}, {})".format(v["data_name"], v["model_name"], v["prepr_name"], v["cv_acc"],
            #                                          "normalized" if v["is_normalized"] == 1 else "unnorm.",
            #                                          "linvar" if v["param_linvar"] else "mean"   ))
            print(v)


        print("****************************")
        print("****************************")
        print("Best feature set for each model")
        best_setup(["data_name", "model_name", "param_linvar"], "cv_acc", ["prepr_name", "cv_acc", "test_acc"])


        print("****************************")
        print("****************************")
        print("Linear seperability for each preprocessing")
        rows = results.loc[(results.model_name=="linear") & (results.param_linvar==False)].index
        for idx in rows:
            row = results.iloc[idx]
            print(row[["data_name", "prepr_name", "cv_acc", "test_acc"]])
            print("--------")



        print("****************************")
        print("****************************")
        print("Benefit of using linvar for each preprocessing and model")
        best_linvar = results.loc[(results.param_linvar==True) & (results.is_normalized==True)].groupby(["data_name", "prepr_name", "model_name"])["cv_acc"].idxmax()
        for idx in best_linvar:
            row = results.iloc[idx]
            cmp = results.loc[(results.param_linvar==False) &
                                  (results.data_name == row["data_name"]) &
                                  (results.model_name == row["model_name"]) &
                                  (results.prepr_name == row["prepr_name"])
                                ].iloc[0]
            print(row[["data_name", "prepr_name", "model_name"]])
            print("cv_acc benefit: ", row["cv_acc"]-cmp["cv_acc"])
            print("test_acc benefit: ", row["test_acc"]-cmp["test_acc"])
            print("--------")


        #print(results)
    elif cmd == "analyze_trained":
        analyze_trained_models()
    elif cmd == "channel_activation":
        visualize_channel_activation()
    elif cmd == "is_normalized":
        file = sys.argv[2]
        value = sys.argv[3]
        results = pd.read_csv(file)
        results["is_normalized"] = value
        results.to_csv(file, index=False)

    else:
        print("""Usage: python {} <cmd>
           with cmd: cv - runs cv for all models/datasets/preprs
                     merge_csv - concates all csvs in results
                     open_csv <path> opens a csv and shows the best row per dataset
                     analyze_trained - plots filters and stuff
                     important_channels - checks the effect of setting a channel to 0 for different models/datasets
                     channel_activation - plots the activation of channels for MIR and Spectro data
                     is_normalized <path to csv> <value> - set the new column 'is_normalized' is <value>, to specify if the file was generated with normalized channels
        """.format(sys.argv[0]))
        print("Unknown cmd: "+cmd)

