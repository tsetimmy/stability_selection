import numpy as np

import sys
sys.path.append('..')

import argparse

from stability_selection import generate_data, stability_selection, plot_toy
from significance_lasso.scale import scale

import matplotlib.pyplot as plt

import uuid
import pickle

def get_area(x, y):
    assert len(x) == len(y)
    assert len(x) >= 2

    total_area = 0.
    for i in range(1, len(x)):
        x0 = x[i - 1]
        x1 = x[i]
        y0 = y[i - 1]
        y1 = y[i]

        #area of square and triangle
        ymin = min(y0, y1)
        total_area += (ymin  + .5 * np.abs(y1 - y0)) * (x1 - x0)

    return total_area

def toy(n_perms=100):
    n = 100*2
    p = 200
    n_bootstraps = 100
    lam_list = np.linspace(.001, .5, 100)

    X, y, b = generate_data(n, p, noise_var=.6)
    X = scale(X)

    freqs_list = stability_selection(X, y, b, n_bootstraps, lam_list, weakness=.2)
    areas = np.array([get_area(lam_list, freq) for freq in freqs_list.T])

    areas_list = [areas]

    y_perm = y.copy()
    for perm in range(n_perms):
        print(perm, 'out of:', n_perms)
        np.random.shuffle(y_perm)
        freqs_list = stability_selection(X, y_perm, b, n_bootstraps, lam_list, weakness=.2)
        areas = np.array([get_area(lam_list, freq) for freq in freqs_list.T])

        areas_list.append(areas)

    results = {'data': 'toy',
               'b': b,
               'n_perms': n_perms,
               'areas_list': areas_list}

    filename = str(uuid.uuid4()) + '_toy.pickle'
    pickle.dump(results, open(filename, 'wb'))

def prostate(n_perms=100):
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

    freqs_list = stability_selection(X_train, y_train, None, n_bootstraps, lam_list, weakness=.2)
    areas = np.array([get_area(lam_list, freq) for freq in freqs_list.T])

    areas_list = [areas]

    y_perm = y_train.copy()

    for perm in range(n_perms):
        print(perm, 'out of:', n_perms)
        np.random.shuffle(y_perm)
        freqs_list = stability_selection(X_train, y_perm, None, n_bootstraps, lam_list, weakness=.2)
        areas = np.array([get_area(lam_list, freq) for freq in freqs_list.T])

        areas_list.append(areas)

    results = {'data': 'prostate',
               'n_perms': n_perms,
               'areas_list': areas_list}
    
    filename = str(uuid.uuid4()) + '_prostate.pickle'
    pickle.dump(results, open(filename, 'wb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, choices=['toy', 'prostate'], default='toy')
    parser.add_argument('--n_perms', type=int, default=100)
    args = parser.parse_args()

    print(sys.argv)
    print(args)

    #eval(args.data+ '()')
    if args.data == 'toy':
        toy(n_perms=args.n_perms)
    elif args.data == 'prostate':
        prostate(n_perms=args.n_perms)
