import numpy as np
from sklearn.linear_model import Lasso
import sys
sys.path.append('..')

import argparse

from utils import unison_shuffled_copies
from significance_lasso.scale import scale

import matplotlib.pyplot as plt

def plot_prostate(lam_list, freqs_list, *unused):
    labels = ['lcavol', 'lweight', 'age', 'lbph', 'svi', 'lcp', 'gleason', 'pgg45']
    for l, f in zip(labels, freqs_list.T):
        plt.plot(lam_list, f, label=l)
    plt.legend()
    plt.grid()
    plt.show()

def plot_toy(lam_list, freqs_list, b):
    plt.plot(lam_list, freqs_list.T[np.where(b == 0.)].T, 'k:', linewidth=.5)
    plt.plot(lam_list, freqs_list.T[np.where(b == 1.)].T, 'r-', linewidth=1.)
    print(np.where(b == 0.))
    print(np.where(b == 1.))
    plt.grid()
    plt.show()

def stability_selection(X, y, b, n_bootstraps, lam_list, weakness=.5, plot=None):
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
    if plot is not None:
        plot(lam_list, freqs_list, b)
    return freqs_list

def generate_data(n, p, noise_var=1.):
    X = np.random.normal(size=[n, p])
    b = np.zeros(p)
    b[:1] = 1.
    np.random.shuffle(b)

    e = np.random.normal(scale=noise_var**.5, size=n)

    y = X @ b + e

    return X, y, b

def toy_example():
    n = 100*2
    p = 200
    n_bootstraps = 100
    lam_list = np.linspace(.001, .5, 100)

    X, y, b = generate_data(n, p, noise_var=.6)
    X = scale(X)

    stability_selection(X, y, b, n_bootstraps, lam_list, weakness=.2, plot=plot_toy)

def prostate_example():
    data = np.loadtxt('../significance_lasso/prostate.data', skiprows=1, usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10))

    train_indices = np.where(data[:, -1] == 1.)[0]
    test_indices = np.where(data[:, -1] == 0.)[0]

    y_train = data[train_indices, 8]
    y_test = data[test_indices, 8]

    global_scale = True

    if global_scale:
        X_scaled = scale(data[:, :8])
        X_train = X_scaled[train_indices, :]
        X_test = X_scaled[test_indices, :]
    else:
        X_train = data[train_indices, :8]
        X_test = data[test_indices, :8]
        X_train = scale(X_train)

    n_bootstraps = 100*2
    lam_list = np.linspace(.001, .5, 100)
    stability_selection(X_train, y_train, None, n_bootstraps, lam_list, weakness=.2, plot=plot_prostate)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--example', type=str, choices=['toy_example', 'prostate_example'], default='toy_example')
    args = parser.parse_args()

    print(sys.argv)
    print(args)

    eval(args.example + '()')

#    n = 100
#    p = 1000
#
#    X = np.random.normal(size=[n, p])
#   
#    
#    normalizer = np.sqrt(np.square(X).sum(axis=0))
#    X_scaled = X / normalizer
#    
#    print(normalizer.shape)
#
#    X = (np.arange(10) + 1.).reshape(2, 5)
#    normalizer = np.sqrt(np.square(X).sum(axis=0))
#    X_scaled = X / normalizer
#
#
#    print(X.shape)
#    print(normalizer.shape)
