import numpy as np
import sys
import pickle
import argparse

import matplotlib.pyplot as plt

def sort_print(labels, pvalues):
    indices = np.argsort(pvalues)
    for a, b in zip(np.array(labels)[indices], pvalues[indices]):
        print(a, b)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str)
    args = parser.parse_args()

    print(sys.argv)
    print(args)

    data = pickle.load(open(args.filename, 'rb'))
    n_perms = data['n_perms']
    if data['data'] == 'prostate':
        areas = data['areas_list']

        truth = areas[0].copy()
        perms = np.stack(areas[1:])

        pvalues = np.zeros_like(truth)
        
        for i in range(len(truth)):
            for j in range(len(perms)):
                if perms[j, i] >= truth[i]:
                    pvalues[i] += 1.
        pvalues /= float(len(perms) + 1)
        labels = ['lcavol', 'lweight', 'age', 'lbph', 'svi', 'lcp', 'gleason', 'pgg45']

        print('n_perms:', n_perms)
        print('--------')
        sort_print(labels, pvalues)
        print('--------')
        annals_pvalues = np.array([0., .052, .65, .929,
                                   .174, .051, .978, .353])
        sort_print(labels, annals_pvalues)
    elif data['data'] == 'toy':
        areas = data['areas_list']

        b = data['b']

        truth = areas[0].copy()
        perms = np.stack(areas[1:])

        pvalues = np.zeros_like(truth)

        for i in range(len(truth)):
            for j in range(len(perms)):
                if perms[j, i] >= truth[i]:
                    pvalues[i] += 1.
        pvalues /= float(len(perms) + 1)

        print('n_perms:', n_perms)
        print('--------')
        for bb, pvalue in zip(b, pvalues):
            print(bb, pvalue)

    # Sort pvalues and b by ordering of former.
    sorted_indices = np.argsort(pvalues)
    pvalues_sorted = pvalues[sorted_indices]
    b_sorted = b[sorted_indices]

    zero_b_indices = np.where(b_sorted == 0.)
    non_zero_b_indices = np.where(b_sorted != 0.)

    theoretical_pvalues = np.linspace(0., 1., len(pvalues_sorted) + 2)[1:-1]

    plt.scatter(theoretical_pvalues[zero_b_indices], pvalues_sorted[zero_b_indices], s=8., label='non zero betas (true)')
    plt.scatter(theoretical_pvalues[non_zero_b_indices], pvalues_sorted[non_zero_b_indices], color='red', s=8., label='zero betas (true)')

    plt.plot([0., theoretical_pvalues.max()], [0., theoretical_pvalues.max()], 'k:', linewidth=.5)
    plt.grid()
    plt.legend()
    plt.title('simulated data')
    plt.xlabel('theoretical pvalues')
    plt.ylabel('empirical pvalues')
    plt.show()

if __name__ == '__main__':
    main()
