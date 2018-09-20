import os
from os import listdir
from os.path import isfile, join

import numpy as np
import matplotlib.pyplot as plt
import madmom

import sys
sys.path.append('../src')
from preprocessing import RhythmData, SpectroData, MIRData, NORMALIZE_CHANNELS
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

import pickle
import pandas as pd
pd.set_option('display.max_columns', 15)

import pdb
import random

MUSIC = 1
SPEECH = 0

RUN_NAME='cluster_cv'

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


model_names =["simple_cnn--linvar", "simple_cnn", "linear--linvar", "linear", "big_cnn--linvar", "big_cnn", "big_linear--linvar", "big_linear", "mean_svm"]

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





def run_on_all(experiment):
    cols = ['data_name','prepr_name','model_name', 'param_c', 'param_gamma', 'param_linvar', 'test_acc', 'test_f1', 'test_acc_pc', 'test_acc_nc', 'cv_acc', 'cv_f1', 'cv_acc_pc', 'cv_acc_nc']
    results = pd.DataFrame(columns=cols)
    for data_name, kwargs in data_path.items():

        if data_name == "columbia-test": continue # don't use the test set for training
        for Preprocessor in [RhythmData, MIRData, SpectroData]:
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
    cmd = sys.argv[1]

    open_csv = None
    if cmd == "cv":
        add_svm_grid()
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
                all.append(add)
            except Exception as e:
                print("Skipped {} bc {} ({})".format(file, e, type(e)))
        merged = pd.concat(all)
        merged.to_csv("results/merged.csv", index=False)
        open_csv = "results/merged.csv"


    if cmd == "open-csv":
        open_csv = sys.argv[2]

    if open_csv:
        save_file_name = open_csv
        results = pd.read_csv(save_file_name)

        results = results.dropna(subset=["cv_acc"])

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


        print("Best run per dataset:")
        best_run_per_dataset = results.groupby("data_name")["cv_acc"].idxmax()  # .apply(np.argmax)
        for row in best_run_per_dataset:
            v = results.iloc[row]
            print("{}: {} on {} with {} ({})".format(v["data_name"], v["model_name"], v["prepr_name"], v["cv_acc"], "normalized" if v["is_normalized"]==1 else "unnorm."))
            print(v)
            print("\n")

        best_run_per_model = results.groupby("model_name")["cv_acc"].idxmax()  # .apply(np.argmax)
        print('Best model: {}'.format(best_overall_run["model_name"]))
        print('(On {})'.format(best_overall_run["prepr_name"]))
        print('Model Selection Accuracy: {}'.format(best_overall_run["cv_acc"]))
        #print(results)
    elif cmd == "analyze_trained":
        analyze_trained_models()
    elif cmd == "important_channels":
        important_channels()
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

