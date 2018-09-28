import pandas as pd
from cv_experiment import *
import sys
import pdb
import numpy as np
from matplotlib import pyplot as plt

"""
- weights von mv linear
- pertubation experiment auf mehr models
- falsch klassifizierte dinger speichern und gucken ob es die gleichen sind
"""


def important_channels_single_setup(model_name, Preprocessor):
    MUSIC = 1
    SPEECH = 0

    row_id = 0 # counter for where to store the results

    for data_name, kwargs in data_path.items():

        # if data_name == "columbia-test": continue  # don't use the test set for training
        if not data_name == "GTZAN": continue
        prepr_name = Preprocessor.__name__
        data = Preprocessor(**kwargs)

        col_test_data = Preprocessor(**data_path["columbia-test"])

        music_sample = random.choice(col_test_data.X[col_test_data.Y == MUSIC])
        speech_sample = random.choice(col_test_data.X[col_test_data.Y != MUSIC])

        print("Col test data \nX: {}, Y: {}".format(col_test_data.X.shape, col_test_data.Y.shape))

        print("\n\n-------\nImportant channels {} on {} - {}".format(model_name, data_name, prepr_name))

        model, Xtrain, Ytrain, Xtest, Ytest = get_trained_model(data, model_name, retrain=True)

        col_test_data.X = Xtest

        def acc(p):
            return np.mean((p>0.5)==(data.Y>0.5))

        def kl_div(y, p):
            y_, p_ = 1-y, 1-p
            return np.sum( y*np.log(y/p) ) + np.sum( y_*np.log(y_/p_) )

        y = model.predict(Xtrain)

        effects = []
        for channel in range(data.X.shape[-1]):
            X_ = np.array(Xtrain)
            X_[:,:,:,channel] = 0
            p = model.predict(X_)
            effects.append(1-np.mean((p>0.5)==(y>0.5)))

        plt.bar(list(range(len(effects))), effects)
        plt.title("{} on {}".format(model_name.split("--")[0], prepr_name))
        plt.savefig("effect-{}-{}.png".format(model_name.split("--")[0], prepr_name))
        plt.clf()

def eval_best_models():
    r = pd.read_csv("results/merged.csv")

    nn = r.loc[(r.model_name=="mv_nn") & (r.data_name=="GTZAN") & (r.prepr_name=="RhythmData")]["cv_acc"].idxmax()
    nn = r.iloc[nn]
    important_channels_single_setup(nn["model_name"]+"--"+nn["hyper_params"], RhythmData)

    svm = r.loc[(r.model_name=="mv_svm") & (r.data_name=="GTZAN") & (r.prepr_name=="RhythmData")]["cv_acc"].idxmax()
    svm = r.iloc[svm]
    important_channels_single_setup(svm["model_name"]+"--"+svm["hyper_params"], RhythmData)

    nn = r.loc[(r.model_name == "mv_nn") & (r.data_name == "GTZAN") & (r.prepr_name == "MIRData")][
        "cv_acc"].idxmax()
    nn = r.iloc[nn]
    important_channels_single_setup(nn["model_name"] + "--" + nn["hyper_params"], MIRData)

    svm = r.loc[(r.model_name == "mv_svm") & (r.data_name == "GTZAN") & (r.prepr_name == "MIRData")][
        "cv_acc"].idxmax()
    svm = r.iloc[svm]
    important_channels_single_setup(svm["model_name"] + "--" + svm["hyper_params"], MIRData)

def eval():
    r = pd.read_csv(sys.argv[1])

    # find best hyper params for svm and nn on gtzan

    best_svm = r.loc[(r.model_name=="mv_svm") & (r.data_name=="GTZAN")]["cv_acc"].idxmax()
    best_svm = "mv_svm--" + r.iloc[best_svm]["hyper_params"]

    best_mv_nn = r.loc[(r.model_name=="mv_nn") & (r.data_name=="GTZAN")]["cv_acc"].idxmax()
    best_mv_nn = "mv_nn--" + r.iloc[best_mv_nn]["hyper_params"]

    model_names = [best_svm, best_mv_nn, "mv_linear", "linear", "simple_cnn"]

    important_channels(model_names)


MIR_names = ["M-spectral-flux", "M-super-flux", "M-complex-flux", "M-spectral_centroid", "M-spectral_bandwidth", "M-spectral_flatness", "M-spectral_rolloff", "M-rmse", "M-zero_crossing_rate"]
Rhythm_names = ["R-SpectralOnsetProcessor", "R-RNNOnsetProcessor", "R-CNNOnsetProcessor", "R-SpectralOnsetProcessor", "R-RNNDownBeatProcessor"]
Rhythm_names += ["R-RNNBeatProcessor-{}".format(i) for i in range(9)]

def read():
    df = pd.read_csv("effect_of_channels.csv")
    # ["model", "channel", "preprocesssing", "effect_acc", "effect_kldiv", "test_acc"]

    #pdb.set_trace()
    # importance for best model


    feature_names = ["M-{}".format(i) for i in range(9)]
    feature_names += ["R-{}".format(i) for i in range(14)]

    # correlation between models: do they use the same features?
    def get_model_vector(model_name):
        v = df.loc[df.model.str.startswith(model_name)].sort(["preprocesssing","channel"])["effect_acc"]
        return np.array(v)


    print("************")
    print("Effects on best model per dataset (test_acc)")
    idx = df["test_acc"].idxmax()
    best_model = df.iloc[idx]["model"]


    plt.bar(list(range(len(feature_names))), get_model_vector(best_model))
    plt.xticks(list(range(len(feature_names))), feature_names)
    plt.savefig("../results/channel_effect-best-model.png")
    plt.clf()

    def importance_on_models(model_names, shortname):
        model_vectors = []
        for model_name in model_names:
            model_vectors.append(
                get_model_vector(model_name)
            )

        mean = np.mean(model_vectors, axis=0)
        std = np.std(model_vectors, axis=0)
        plt.bar(list(range(len(feature_names))), mean, label="Mean")
        plt.bar(list(range(len(feature_names))), std, label="Std dev")
        plt.xticks(list(range(len(feature_names))), feature_names)
        plt.legend()
        plt.savefig("../results/channel_effect-{}.png".format(shortname))
        plt.clf()

    # for all models:
    print("Models", df["model"].unique())
    importance_on_models(df["model"].unique(), "all")
    importance_on_models(["mv_nn", "mv_svm", "mv_linear"], "mv-models")
    importance_on_models(["simple_cnn", "linear"], "conv-models")


def mv_linear_weight_plot():

    for data_name, kwargs in data_path.items():

        # if data_name == "columbia-test": continue  # don't use the test set for training
        if not data_name == "GTZAN": continue
        for Preprocessor in [RhythmData, MIRData]:
            prepr_name = Preprocessor.__name__
            data = Preprocessor(**kwargs)
            model, X, _, _, _ = get_trained_model(data, "mv_linear", retrain=True)

            w = np.squeeze(model.model.layers[0].get_weights()[0])
            weights = np.reshape(w, [-1, 2])
            w_mean = weights[:,0]
            w_var = weights[:,1]
            X_mean = np.squeeze(np.mean(X, axis=(0,1,2,)))
            r_mean = w_mean


            plt.bar(list(range(len(r_mean))), r_mean, label="Mean")
            plt.bar(list(range(len(r_mean))), w_var, label="Var", alpha=0.5)
            plt.title(prepr_name)
            plt.legend()
            plt.savefig("weights-mv-linear-{}.png".format(prepr_name))
            plt.clf()


#mv_linear_weight_plot()
eval_best_models()

if False:
    if __name__ == "__main__":
        if len(sys.argv) == 2:
            eval()
        read()



