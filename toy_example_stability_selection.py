import numpy as np

import sys
sys.path.append('..')

import argparse

from stability_selection import stability_selection
from significance_lasso.scale import scale

import matplotlib.pyplot as plt

def plot_toy(lam_list, freqs_list, b):
    zeros = np.where(b == 0.)
    non_zeros = np.where(b == 1.)
    if zeros[0].size > 0:
        plt.plot(lam_list, freqs_list.T[zeros].T, 'k:', linewidth=.5)
    if non_zeros[0].size > 0:
        plt.plot(lam_list, freqs_list.T[non_zeros].T, 'r-', linewidth=1.)
    plt.xlabel('lambdas')
    plt.ylabel('frequency')
    plt.grid()
    plt.show()

def generate_data(n, p, n_b=1, noise_var=1.):
    X = np.random.normal(size=[n, p])
    b = np.zeros(p)
    b[:n_b] = 1.
    np.random.shuffle(b)

    e = np.random.normal(scale=noise_var**.5, size=n)

    y = X @ b + e

    return X, y, b

def toy_example(n=200, p=200, n_bootstraps=100,
                lam_low=.001, lam_high=.5, n_lams=100,
                n_b=1, noise_var=.6, weakness=.2):
    lam_list = np.linspace(lam_low, lam_high, n_lams)

    X, y, b = generate_data(n, p, n_b=n_b, noise_var=noise_var)
    X = scale(X)

    freqs_list = stability_selection(X, y, b, n_bootstraps, lam_list, weakness=weakness)

    plot_toy(lam_list, freqs_list, b)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--n', type=int, default=200)
    parser.add_argument('--p', type=int, default=200)
    parser.add_argument('--n_bootstraps', type=int, default=100)
    parser.add_argument('--lam_low', type=float, default=.001)
    parser.add_argument('--lam_high', type=float, default=.5)
    parser.add_argument('--n_lams', type=int, default=100)
    parser.add_argument('--n_b', type=int, default=1)
    parser.add_argument('--noise_var', type=float, default=.6)
    parser.add_argument('--weakness', type=float, default=.2)

    args = parser.parse_args()

    print(sys.argv)
    print(args)

    toy_example(n=args.n,
                p=args.p,
                n_bootstraps=args.n_bootstraps,
                lam_low=args.lam_low,
                lam_high=args.lam_high,
                n_lams=args.n_lams,
                n_b=args.n_b,
                noise_var=args.noise_var,
                weakness=args.weakness)
