import numpy as np

NORMALIZE_CHANNELS=True

class LameModelException(Exception):
    def __init__(self, cv_acc="bad"):
        super().__init__("Lame model spotted: cv_acc:{}".format(cv_acc))

def cv(X, y, method, train_fun, nfolds=10, nrepetitions=5, shuffle=True, norm_channels=NORMALIZE_CHANNELS):
        evals = []
        # evals_reinit = []
        N = X.shape[0]
        for rep in range(nrepetitions):

                print('......begin iteration {}/{}'.format(rep+1, nrepetitions))

                # shuffle data
                if shuffle:
                        I = np.random.permutation(N)
                else:
                        I = np.arange(N)

                fold_size = N // nfolds
                fold_rest = N %  nfolds

                # FOLD
                for f_idx in range(nfolds):

                        # split in train and test folds
                        if f_idx < fold_rest:
                                start_idx = (fold_size + 1) * f_idx
                                stop_idx  = (fold_size + 1) * (f_idx + 1)
                        else:
                                start_idx = (fold_size + 1) * fold_rest + fold_size * (f_idx - fold_rest)
                                stop_idx  = (fold_size + 1) * fold_rest + fold_size * (f_idx - fold_rest + 1)
                        
                        train_indices = np.ones(N, dtype='bool')
                        train_indices[start_idx:stop_idx] = False
                        val_indices = np.logical_not(train_indices)

                        # slices are views, we dont want the original array to change
                        if norm_channels:
                            X_cp = X.copy() 
                        else:
                            X_cp = X

                        X_train = X_cp[I[train_indices]]
                        y_train = y[I[train_indices]]   
                        X_val   = X_cp[I[val_indices]]
                        y_val   = y[I[val_indices]]

                        if norm_channels:
                            X_train, stddev = normalize_channels(X_train)
                            X_val /= stddev

                        model = method()
                        
                        # reinit_evaluation=model.evaluate(X_val, y_val, verbose=0)
                        # evals_reinit.append(reinit_evaluation)

                        train_fun(model, X_train, y_train)
                        evaluation = model.evaluate(X_val, y_val, verbose=0)
                        if len(evaluation)==5:
                            evaluation=evaluation[1:]
                        if evaluation[0] < 0.9:
                            raise LameModelException(evaluation[0])

                        evals.append(evaluation)
                        print('cv: finished fold {}/{}'.format(f_idx+1, nfolds))
                        print('(score was {})'.format(evaluation))


        return [sum(y) / len(y) for y in zip(*evals)] #, np.mean(evals_reinit, axis=0)

def normalize_channels(X, unbiased=False):
    # divide each channel by its std dev
    if unbiased:
        ddof = 1
    else:
        ddof = 0

    stddev = np.std(X, axis=(0,1,2), keepdims=True, ddof=ddof)
    X /= stddev
    return X, stddev
