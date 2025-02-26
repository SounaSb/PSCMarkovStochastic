import numpy as np

def circular_mean(arr, w):
    result = np.zeros_like(arr)
    for i in range(len(arr)):
        
        result[i] = np.mean(np.roll(arr, -i+2)[:w])
    return result


# C^inf function which is null on negative numbers
def exp_cinf(x):
    return np.exp(-1 / np.maximum(x, 1e-18)) * np.int64(x > 0)

# Plateau function (support = [0.2, 0.3])
def plateau(x):
    return (30 * exp_cinf(x - .2) * exp_cinf(-x + .3))**2

