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

MIR_names = ["spectral\nflux", "super\nflux", "complex\nflux", "spectral\ncentroid", "spectral\nbandwidth", "spectral\nflatness", "spectral\nrolloff", "rmse", "0-cross\nrate"]
Rhythm_names = ["Spectral\nOnset", "RNNOnset", "CNNOnset", "SpectralOnset", "RNNDownBeat"]
Rhythm_names += ["RNNBeat-{}".format(i) for i in range(9)]


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
            X_[:,:,:,channel] = np.mean(X_[:,:,:,channel])
            p = model.predict(X_)
            effects.append(1-np.mean((p>0.5)==(y>0.5)))

        mname = model_name.split("--")[0]
        color = "g" if mname == "mv_nn" else "r"
        ticks = np.array(list(range(len(effects))), dtype=np.float64)
        if mname == "mv_nn":
            pos =  ticks + 0.2
        else:
            pos=ticks

        if prepr_name == "RhythmData":
            feature_names = Rhythm_names
        else:
            feature_names = MIR_names

        plt.bar(pos, effects, alpha=0.5, label=mname, color=color, width=0.4)
        plt.title(prepr_name)

        plt.xticks(ticks, feature_names, rotation='vertical')

        return data



def eval_best_models():
    r = pd.read_csv("results/merged.csv")

    plt.figure(figsize=(6, 4))
    #plt.subplot(211)
    nn = r.loc[(r.model_name=="mv_nn") & (r.data_name=="GTZAN") & (r.prepr_name=="RhythmData")]["cv_acc"].idxmax()
    nn = r.iloc[nn]
    important_channels_single_setup(nn["model_name"]+"--"+nn["hyper_params"], RhythmData)

    svm = r.loc[(r.model_name=="mv_svm") & (r.data_name=="GTZAN") & (r.prepr_name=="RhythmData")]["cv_acc"].idxmax()
    svm = r.iloc[svm]
    data = important_channels_single_setup(svm["model_name"]+"--"+svm["hyper_params"], RhythmData)
    plt.legend()

    #ax = plt.subplot(212)
    #mv_linear_weight_plot(data, RhythmData)
    #ax.xaxis.tick_top()

    plt.tight_layout()
    plt.savefig("importance-rhythm.png")
    plt.clf()

    plt.figure(figsize=(6, 4))
    #plt.subplot(211)
    nn = r.loc[(r.model_name == "mv_nn") & (r.data_name == "GTZAN") & (r.prepr_name == "MIRData")][
        "cv_acc"].idxmax()
    nn = r.iloc[nn]
    important_channels_single_setup(nn["model_name"] + "--" + nn["hyper_params"], MIRData)

    svm = r.loc[(r.model_name == "mv_svm") & (r.data_name == "GTZAN") & (r.prepr_name == "MIRData")][
        "cv_acc"].idxmax()
    svm = r.iloc[svm]
    data = important_channels_single_setup(svm["model_name"] + "--" + svm["hyper_params"], MIRData)
    plt.legend()

    #ax = plt.subplot(212)
    #mv_linear_weight_plot(data, MIRData)
    #ax.xaxis.tick_top()

    plt.tight_layout()
    plt.savefig("importance-mir.png")
    plt.clf()

def eval():
    r = pd.read_csv(sys.argv[1])

    # find best hyper params for svm and nn on gtzan

    best_svm = r.loc[(r.model_name=="mv_svm") & (r.data_name=="GTZAN")]["cv_acc"].idxmax()
    best_svm = "mv_svm--" + r.iloc[best_svm]["hyper_params"]

    best_mv_nn = r.loc[(r.model_name=="mv_nn") & (r.data_name=="GTZAN")]["cv_acc"].idxmax()
    best_mv_nn = "mv_nn--" + r.iloc[best_mv_nn]["hyper_params"]

    model_names = [best_svm, best_mv_nn, "mv_linear", "linear", "simple_cnn"]

    important_channels(model_names)


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


def mv_linear_weight_plot(data, Preprocessor):

    prepr_name = Preprocessor.__name__
    model, X, _, _, _ = get_trained_model(data, "mv_linear", retrain=True)

    w = np.squeeze(model.model.layers[0].get_weights()[0])
    weights = np.reshape(w, [-1, 2])
    w_mean = weights[:,0]
    w_var = weights[:,1]
    X_mean = np.squeeze(np.mean(X, axis=(0,1,2,)))
    r_mean = w_mean

    r_var = np.where((r_mean>0)==(w_var>0), r_mean+w_var, w_var)
    ticks = list(range(len(r_mean)))
    plt.bar(ticks, w_var, label="Var", width=0.4)
    plt.bar(ticks, r_mean, label="Mean", width=0.4)
    plt.plot([0, len(ticks)-1], [0,0], color="black", linewidth=0.5)
    plt.xticks(ticks, [""]*len(ticks))
    plt.legend()


#mv_linear_weight_plot()
eval_best_models()

if False:
    if __name__ == "__main__":
        if len(sys.argv) == 2:
            eval()
        read()



