import numpy as np

def cv(X, y, method, train_fun, evaluation=accuracy, nfolds=10, nrepetitions=5, shuffle=True):
        evals = []
        # evals_reinit = []
        N = X.shape[0]
        for rep in range(nrepetitions):

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

                        X_train = X[I[train_indices]]
                        y_train = y[I[train_indices]]   
                        X_val   = X[I[val_indices]]
                        y_val   = y[I[val_indices]]

                        model = method()
                        
                        # reinit_evaluation=model.evaluate(X_val, y_val, verbose=0)
                        # evals_reinit.append(reinit_evaluation)

                        train_fun(model, X_train, y_train)
                        evaluation = model.evaluate(X_val, y_val, verbose=0)

                        evals.append(evaluation)

        return np.mean(evals, axis=0) #, np.mean(evals_reinit, axis=0)
