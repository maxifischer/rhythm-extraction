import pandas as pd
from cv_experiment import important_channels
import sys
import pdb
import numpy as np
from matplotlib import pyplot as plt


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
    print("************")
    print("Effects on best model per dataset (test_acc)")
    idx = df["test_acc"].idxmax()
    best_model = df.iloc[idx]["model"]

    # correlation between models: do they use the same features?
    def get_model_vector(model_name):
        v = df.loc[df.model == model_name].sort(["preprocesssing","channel"])["effect_acc"]
        return np.array(v)

    model_names = df["model"].unique()
    model_vectors = []
    for model_name in model_names:
        model_vectors.append(
            get_model_vector(model_name)
        )

    print("Model vectors", model_vectors)

    """
    106     GTZAN  simple_cnn      0.0        MIRData    0.179688     20.649775   
    107     GTZAN  simple_cnn      1.0        MIRData    0.281250     45.676178   
    108     GTZAN  simple_cnn      2.0        MIRData    0.398438     78.654411   
    109     GTZAN  simple_cnn      3.0        MIRData    0.484375    224.813049   
    110     GTZAN  simple_cnn      4.0        MIRData    0.023438      2.745762   
    111     GTZAN  simple_cnn      5.0        MIRData    0.085938      6.730135   
    112     GTZAN  simple_cnn      6.0        MIRData    0.359375    100.214455   
    113     GTZAN  simple_cnn      7.0        MIRData    0.515625    318.423004   
    114     GTZAN  simple_cnn      8.0        MIRData    0.187500     26.169098   
    56      GTZAN  simple_cnn      0.0     RhythmData    0.289062     63.214935   
    57      GTZAN  simple_cnn      1.0     RhythmData    0.125000     23.657265   
    58      GTZAN  simple_cnn      2.0     RhythmData    0.117188     10.050459   
    59      GTZAN  simple_cnn      3.0     RhythmData    0.507812           inf   
    60      GTZAN  simple_cnn      4.0     RhythmData    0.132812      9.475353   
    61      GTZAN  simple_cnn      5.0     RhythmData    0.031250      1.505660   
    62      GTZAN  simple_cnn      6.0     RhythmData    0.234375     28.541260   
    63      GTZAN  simple_cnn      7.0     RhythmData    0.023438      1.042080   
    64      GTZAN  simple_cnn      8.0     RhythmData    0.117188      7.979902   
    65      GTZAN  simple_cnn      9.0     RhythmData    0.031250      2.577069   
    66      GTZAN  simple_cnn     10.0     RhythmData    0.304688           inf   
    67      GTZAN  simple_cnn     11.0     RhythmData    0.156250     30.833712   
    68      GTZAN  simple_cnn     12.0     RhythmData    0.007812      0.733049   
    69      GTZAN  simple_cnn     13.0     RhythmData    0.218750     23.436228   

    """

    feature_names = ["R-spectral-flux", "R-super-flux", "R-complex-flux", "M-spectral_centroid", "M-spectral_bandwidth", "M-spectral_flatness", "M-spectral_rolloff", "M-rmse", "M-zero_crossing_rate"]
    feature_names += ["R-SpectralOnsetProcessor", "R-RNNOnsetProcessor", "R-CNNOnsetProcessor", "R-SpectralOnsetProcessor", "R-RNNDownBeatProcessor"]
    feature_names += ["R-RNNBeatProcessor-{}".format(i) for i in range(9)]

    feature_names = ["M-{}".format(i) for i in range(9)]
    feature_names += ["R-{}".format(i) for i in range(14)]


    plt.bar(list(range(len(feature_names))), get_model_vector(best_model))
    plt.xticks(list(range(len(feature_names))), feature_names)
    plt.savefig("../results/channel_effect-best-model.png")
    plt.clf()

    mean = np.mean(model_vectors, axis=0)
    var = np.var(model_vectors, axis=0)
    plt.bar(list(range(len(feature_names))), mean, label="Mean")
    plt.bar(list(range(len(feature_names))), var, label="Var")
    plt.xticks(list(range(len(feature_names))), feature_names)
    plt.legend()
    plt.savefig("../results/channel_effect-mean_and_var.png")
    plt.clf()


    # channel importance averaged over models: are some features always important


    # has any model a high value in importance of channel - avg importance of that channel?
    #    this models makes extra good use of this feature


if __name__ == "__main__":
    if len(sys.argv) == 2:
        eval()
    read()



