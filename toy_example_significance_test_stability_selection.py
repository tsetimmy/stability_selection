import numpy as np

import sys
sys.path.append('..')

import argparse

from stability_selection import stability_selection
from toy_example_stability_selection import generate_data
from significance_lasso.scale import scale

from utils import get_area
import datetime

import matplotlib.pyplot as plt

import uuid
import pickle

def toy(n=200, p=200, n_bootstraps=100,
        n_perms=100, lam_low=.001, lam_high=.5,
        n_b=1, n_lams=100, noise_var=.6,
        weakness=.2):
    lam_list = np.linspace(lam_low, lam_high, n_lams)

    X, y, b = generate_data(n, p, n_b=n_b, noise_var=noise_var)
    X = scale(X)

    freqs_list = stability_selection(X, y, b, n_bootstraps, lam_list, weakness=weakness)
    areas = np.array([get_area(lam_list, freq) for freq in freqs_list.T])

    areas_list = [areas]

    y_perm = y.copy()
    for perm in range(n_perms):
        print(perm, 'out of:', n_perms)
        np.random.shuffle(y_perm)
        freqs_list = stability_selection(X, y_perm, b, n_bootstraps, lam_list, weakness=weakness)
        areas = np.array([get_area(lam_list, freq) for freq in freqs_list.T])

        areas_list.append(areas)

    results = {'data': 'toy',
               'b': b,
               'n_b': n_b,
               'n_perms': n_perms,
               'weakness': weakness,
               'areas_list': areas_list}

    filename = str(datetime.datetime.now()).replace(' ', ',') +\
               '_' + str(uuid.uuid4()) + '_toy.pickle'
    pickle.dump(results, open(filename, 'wb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=200)
    parser.add_argument('--p', type=int, default=200)
    parser.add_argument('--n_bootstraps', type=int, default=100)
    parser.add_argument('--n_perms', type=int, default=100)
    parser.add_argument('--lam_low', type=float, default=.001)
    parser.add_argument('--lam_high', type=float, default=.5)
    parser.add_argument('--n_lams', type=int, default=100)
    parser.add_argument('--n_b', type=int, default=1)
    parser.add_argument('--noise_var', type=float, default=.6)
    parser.add_argument('--weakness', type=float, default=.2)
    args = parser.parse_args()

    print(sys.argv)
    print(args)

    toy(n=args.n,
        p=args.p,
        n_bootstraps=args.n_bootstraps,
        n_perms=args.n_perms,
        lam_low=args.lam_low,
        lam_high=args.lam_high,
        n_lams=args.n_lams,
        n_b=args.n_b,
        noise_var=args.noise_var,
        weakness=args.weakness)
