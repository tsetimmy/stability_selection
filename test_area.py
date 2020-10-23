import numpy as np

from sklearn.linear_model import Lasso

import sys
sys.path.append('..')

import argparse
from toy_example_stability_selection import generate_data, plot_toy
from significance_lasso.scale import scale

from stability_selection import stability_selection

import matplotlib.pyplot as plt

def generate_data2(n, p, n_b=1, noise_var=1.):
    X = np.random.normal(size=[n, p])
    b = np.zeros(p)
    b[0] = 1.

    e = np.random.normal(scale=noise_var**.5, size=n)

    y = X @ b + e
    
    y_perm = y.copy()
    np.random.shuffle(y_perm)

    return X, y, b, e, y_perm

def test_area(n=2, p=2, lam_low=.001,
              n_bootstraps=100, lam_high=.5, n_lams=100,
              n_b=1, noise_var=.6):

    lam_list = np.linspace(lam_low, lam_high, n_lams)

    freqs = np.zeros([n_lams, p])
    freqs2 = np.zeros([n_lams, p])
    freqs3 = np.zeros([n_lams, p])
    for i in range(n_bootstraps):
        print(i, 'of', n_bootstraps)
        X, y, b, y2, y3 = generate_data2(n, p, n_b, noise_var)
        X = scale(X)
        b_hat = []
        b_hat2 = []
        b_hat3 = []
        for lam in lam_list:
            clf = Lasso(alpha=lam, max_iter=5000)
            clf.fit(X, y)
            b_hat.append(clf.coef_)

            clf2 = Lasso(alpha=lam, max_iter=5000)
            clf2.fit(X, y2)
            b_hat2.append(clf2.coef_)

            clf3 = Lasso(alpha=lam, max_iter=5000)
            clf3.fit(X, y3)
            b_hat3.append(clf3.coef_)

        b_hat = np.stack(b_hat, axis=0)
        b_hat2 = np.stack(b_hat2, axis=0)
        b_hat3 = np.stack(b_hat3, axis=0)

        freqs += (np.abs(b_hat) > 0.).astype(np.float64)
        freqs2 += (np.abs(b_hat2) > 0.).astype(np.float64)
        freqs3 += (np.abs(b_hat3) > 0.).astype(np.float64)

    freqs /= float(n_bootstraps)
    freqs2 /= float(n_bootstraps)
    freqs3 /= float(n_bootstraps)

    plt.figure()
    plt.plot(lam_list, freqs[:, 1], label='beta = 0 (original)', color='red', marker='o')
    plt.plot(lam_list, freqs[:, 0], label='beta != 0 (original)', color='red', marker='x')

    plt.plot(lam_list, freqs2[:, 1], label='beta = 0 (noise)', color='blue', marker='o')
    plt.plot(lam_list, freqs2[:, 0], label='beta != 0 (noise)', color='blue', marker='x')

    plt.plot(lam_list, freqs3[:, 1], label='beta = 0 (permuted)', color='green', marker='o')
    plt.plot(lam_list, freqs3[:, 0], label='beta != 0 (permuted)', color='green', marker='x')


    plt.xlabel('lambdas')
    plt.ylabel('frequency')
    plt.grid()
    plt.legend()

    #plt.savefig('test_area.jpg')
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=2)
    parser.add_argument('--p', type=int, default=2)
    parser.add_argument('--n_bootstraps', type=int, default=100)
    parser.add_argument('--lam_low', type=float, default=.001)
    parser.add_argument('--lam_high', type=float, default=.5)
    parser.add_argument('--n_lams', type=int, default=100)
    parser.add_argument('--n_b', type=int, default=1)
    parser.add_argument('--noise_var', type=float, default=.6)
    args = parser.parse_args()

    print(sys.argv)
    print(args)

    test_area(n=args.n,
              p=args.p,
              n_bootstraps=args.n_bootstraps,
              lam_low=args.lam_low,
              lam_high=args.lam_high,
              n_lams=args.n_lams,
              n_b=args.n_b,
              noise_var=args.noise_var)

def diagnostic():
    n = 2000
    p = 2
    n_b = 2
    noise_var = .01
    X, y, b, y2, y3 = generate_data2(n, p, n_b, noise_var)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], y2)
    plt.xlabel('x0')
    plt.ylabel('y')
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 1], y2)
    plt.xlabel('x1')
    #plt.ylabel('y')
    plt.grid()
    plt.suptitle('noise')
    #plt.tight_layout()
    #plt.show()

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], y)
    plt.xlabel('x0')
    plt.ylabel('y')
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 1], y)
    plt.xlabel('x1')
    #plt.ylabel('y')
    plt.grid()
    plt.suptitle('original')
    #plt.tight_layout()
    #plt.show()

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], y3)
    plt.xlabel('x0')
    plt.ylabel('y')
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 1], y3)
    plt.xlabel('x1')
    #plt.ylabel('y')
    plt.grid()
    plt.suptitle('permuted')
    #plt.tight_layout()
    plt.show()


def diagnostic2():
    n = 200
    p = 2
    n_b = 2
    noise_var = .01

    lam = 0.0060404


    n_bootstraps = 100
    b_hat = []
    b_hat2 = []
    b_hat3 = []
    for _ in range(n_bootstraps):
        X, y, b, y2, y3 = generate_data2(n, p, n_b, noise_var)

        clf = Lasso(alpha=lam, max_iter=5000)
        clf.fit(X, y)
        b_hat.append(clf.coef_)

        clf2 = Lasso(alpha=lam, max_iter=5000)
        clf2.fit(X, y2)
        b_hat2.append(clf2.coef_)

        clf3 = Lasso(alpha=lam, max_iter=5000)
        clf3.fit(X, y3)
        b_hat3.append(clf3.coef_)
    b_hat = np.array(b_hat)
    b_hat2 = np.array(b_hat2)
    b_hat3 = np.array(b_hat3)

    import seaborn as sns
    plt.figure()
    plt.subplot(1, 2, 1)
    sns.distplot(b_hat[:, 0], label='original')
    sns.distplot(b_hat2[:, 0], label='noise')
    sns.distplot(b_hat3[:, 0], label='permuted')
    plt.xlabel('x0')
    plt.ylabel('frequency')
    plt.legend()
    plt.grid()


    plt.subplot(1, 2, 2)
    sns.distplot(b_hat[:, 1], label='original')
    sns.distplot(b_hat2[:, 1], label='noise')
    sns.distplot(b_hat3[:, 1], label='permuted')
    plt.xlabel('x1')
    plt.grid()

    plt.suptitle('frequency of b_hat')


    #plt.show()
    plt.savefig('test_area_hist.jpg')

    print('mean (non-zero) frequencies')
    print('original')
    print((np.abs(b_hat) > 0.).astype(np.float64).mean(axis=0))
    print('noise')
    print((np.abs(b_hat2) > 0.).astype(np.float64).mean(axis=0))
    print('permuted')
    print((np.abs(b_hat3) > 0.).astype(np.float64).mean(axis=0))

def diagnostic3():
    n = 200
    p = 1
    noise_var = .01

    lam = 0.0060404

    n_bootstraps = 100

    b_hat2 = []
    b_hat3 = []
    for _ in range(n_bootstraps):
        X, _, _, y2, y3 = generate_data2(n, p, noise_var=noise_var)

        plt.figure()
        plt.scatter(X, y2, label='noise')
        plt.scatter(X, y3, label='permuted')
        plt.legend()
        plt.grid()
        plt.show()


        clf2 = Lasso(alpha=lam, max_iter=5000)
        clf2.fit(X, y2)
        b_hat2.append(clf2.coef_)

        clf3 = Lasso(alpha=lam, max_iter=5000)
        clf3.fit(X, y3)
        b_hat3.append(clf3.coef_)
    b_hat2 = np.array(b_hat2)
    b_hat3 = np.array(b_hat3)

    import seaborn as sns
    plt.figure()
    sns.distplot(b_hat2, label='noise')
    sns.distplot(b_hat3, label='permuted')
    plt.xlabel('x')
    plt.ylabel('frequency')
    plt.legend()
    plt.grid()

    plt.show()

if __name__ == '__main__':
    #main()
    #diagnostic()
    #diagnostic2()
    diagnostic3()
