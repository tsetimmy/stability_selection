import numpy as np

import sys
sys.path.append('..')

import argparse

from stability_selection import stability_selection
from utils import load_prostate

import matplotlib.pyplot as plt

def plot_prostate(lam_list, freqs_list):
    labels = ['lcavol', 'lweight', 'age', 'lbph', 'svi', 'lcp', 'gleason', 'pgg45']
    for l, f in zip(labels, freqs_list.T):
        plt.plot(lam_list, f, label=l)
    plt.legend()
    plt.grid()
    plt.show()

def prostate_example(n_bootstraps=100, lam_low=.001, lam_high=.5,
                     n_lams=100, weakness=.2):
    X_train, y_train, X_test, y_test = load_prostate()

    lam_list = np.linspace(lam_low, lam_high, n_lams)
    freqs_list = stability_selection(X_train, y_train, None, n_bootstraps, lam_list, weakness=weakness)
    plot_prostate(lam_list, freqs_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_bootstraps', type=int, default=100)
    parser.add_argument('--lam_low', type=float, default=.001)
    parser.add_argument('--lam_high', type=float, default=.5)
    parser.add_argument('--n_lams', type=int, default=100)
    parser.add_argument('--weakness', type=float, default=.2)

    args = parser.parse_args()

    print(sys.argv)
    print(args)

    prostate_example(n_bootstraps=args.n_bootstraps,
                     lam_low=args.lam_low,
                     lam_high=args.lam_high,
                     n_lams=args.n_lams,
                     weakness=args.weakness)
