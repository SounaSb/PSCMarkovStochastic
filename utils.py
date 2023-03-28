import numpy as np

def circular_mean(arr, w):
    n = len(arr)
    result = np.zeros(n)
    for i in range(n):
        result[i] = np.mean(np.roll(arr, i)[:w])
    return result
