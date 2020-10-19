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
