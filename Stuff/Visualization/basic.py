import numpy as np

def to_onehot(arr:np.ndarray) -> np.ndarray:
    _, x, y, z = arr.shape
    print(arr.shape)
    print()
    out = np.zeros([x, y, z])
    for i, label in enumerate(arr[1:]):
        print(label.shape, i, label.sum())
        out[label==1] = i+1
    return out

def scale_to_one(arr:np.ndarray) -> np.ndarray:
    arr = arr - arr.min()
    arr = arr / arr.max()
    return arr
