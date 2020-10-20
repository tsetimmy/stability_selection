import numpy as np

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

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

def load_prostate(global_scale=True):
    from significance_lasso.scale import scale
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

    return X_train, y_train, X_test, y_test
