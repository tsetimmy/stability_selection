import numpy as np
from sklearn.linear_model import Lasso
from utils import unison_shuffled_copies

def stability_selection(X, y, b, n_bootstraps, lam_list, weakness=.5):
    n, p = X.shape
    X_perm = X.copy()
    y_perm = y.copy()
    m = int(np.floor(n / 2.))
    freqs_list = []
    for lam in lam_list:
        freqs = np.zeros(p)
        for _ in range(n_bootstraps):
            X_perm, y_perm = unison_shuffled_copies(X_perm, y_perm)

            weights = 1. - (1. - weakness) * np.random.randint(2, size=p)

            clf = Lasso(alpha=lam, max_iter=5000)
            clf.fit(weights * X_perm[:m], y_perm[:m])

            non_zeros = (np.abs(clf.coef_) > 0.).astype(np.float64)

            freqs += non_zeros
        freqs /= float(n_bootstraps)
        freqs_list.append(freqs)

    freqs_list = np.stack(freqs_list)
    lam_list = np.array(lam_list)
    return freqs_list
