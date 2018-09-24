import pandas as pd
from cv_experiment import important_channels
import sys

def eval():
    r = pd.read(sys.argv[1])

    # find best hyper params for svm and nn on gtzan

    best_svm = r.loc[(r.model_name=="mv_svm") & (r.data_name="GTZAN")]["cv_acc"].idxmax()
    best_svm = "mv_svm--" + r.iloc[best_svm]["hyper_params"]

    best_svm = r.loc[(r.model_name=="mv_nn") & (r.data_name="GTZAN")]["cv_acc"].idxmax()
    best_svm = "mv_nn--" + r.iloc[best_svm]["hyper_params"]

    model_names = [best_svm, best_mv_nn, "mv_linear", "linear", "simple_cnn"]

    important_channels(model_names)



def read():
    df = pd.read_csv("effect_of_channels.py")

    models =
    datasets =


    # importance for best model
    print("************")
    print("Effects on best model per dataset (test_acc)")
    rows = df.group_by("data_name")["test_acc"].idxmax()
    for idx in rows:
        print(df.iloc[idx])





if __name__ == "__main__":
    if len(sys.argv) == 2:
        eval()
    read()



# - importance for best model per dataset


- correlation between models: do they use the same features?
- channel importance averaged over models: are some features always important
- has any model a high value in importance of channel - avg importance of that channel?
    this models makes extra good use of this feature