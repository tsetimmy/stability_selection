import numpy as np

import sys
sys.path.append('..')

import argparse

from stability_selection import stability_selection
from significance_lasso.scale import scale

from utils import get_area

def prostate(n_bootstraps=100, n_perms=100, lam_low=.001,
             lam_high=.5, n_lams=100, weakness=.2):
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

    lam_list = np.linspace(lam_low, lam_high, n_lams)

    freqs_list = stability_selection(X_train, y_train, None, n_bootstraps, lam_list, weakness=weakness)
    areas = np.array([get_area(lam_list, freq) for freq in freqs_list.T])

    areas_list = [areas]

    y_perm = y_train.copy()

    for perm in range(n_perms):
        print(perm, 'out of:', n_perms)
        np.random.shuffle(y_perm)
        freqs_list = stability_selection(X_train, y_perm, None, n_bootstraps, lam_list, weakness=weakness)
        areas = np.array([get_area(lam_list, freq) for freq in freqs_list.T])

        areas_list.append(areas)

    results = {'data': 'prostate',
               'n_perms': n_perms,
               'areas_list': areas_list}
    
    filename = str(uuid.uuid4()) + '_prostate.pickle'
    pickle.dump(results, open(filename, 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_bootstraps', type=int, default=100)
    parser.add_argument('--n_perms', type=int, default=100)
    parser.add_argument('--lam_low', type=float, default=.001)
    parser.add_argument('--lam_high', type=float, default=.5)
    parser.add_argument('--n_lams', type=int, default=100)
    parser.add_argument('--weakness', type=float, default=.2)

    args = parser.parse_args()

    print(sys.argv)
    print(args)

    prostate(n_bootstraps=args.n_bootstraps,
             n_perms=args.n_perms,
             lam_low=args.lam_low,
             lam_high=args.lam_high,
             n_lams=args.n_lams,
             weakness=args.weakness)
