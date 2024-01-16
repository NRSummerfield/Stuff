# Python inherent
from typing import Union, Optional
import numpy as np
import math as m
import scipy.ndimage as ndimg
import SimpleITK as sitk

def ensure_shape(arr: np.ndarray, targ_shape: list[int], return_changes: bool = False) -> Union[np.ndarray, tuple[np.ndarray, dict]]:
    """
    Function that takes in an array and ensures that it matches the target shape by either cropping or padding each axis.
    NOTE: Assumes that the outer elements are irrelevant to the output and can be sacrificed as needed
    ---
    Args:
    * `arr`: `np.ndarray` full input array that is to be morphed
    * `targ_shape`: `list[int]` target shape that `arr` needs to match 
    * `return_changes`: `bool` to toggle whether to return the changes or not
    ---
    Returns:
    * `np.ndarray` derived from `arr` that has been cropped / padded 
    - OR -
    * `tuple[np.ndarray, dict]` where the `np.ndarray` is the same as above and `dict` holds the changes
    """
    current_shape = list(arr.shape)
    assert len(current_shape) == len(targ_shape)

    padding = []
    changes = {}
    for i, (c, t) in enumerate(zip(current_shape, targ_shape)):
        dif = t - c
        bot = m.floor(abs(dif) / 2)
        top = m.ceil(abs(dif) / 2)
        if dif < 0: # if its a negative, needs to be REDUCED
            _slice = tuple([slice(None) for _ in range(i)] + [slice(bot, -top, 1)])
            arr = arr[_slice]
            padding.append([0,0])
            changes[i] = ['pad', bot, top] # tracking the opposite
        
        elif dif > 0: # if its positive, needs to be PADDED
            padding.append([bot, top])
            changes[i] = ['crop', bot, top] # tracking the opposite
        
        else: # if its exact, needs to be ignored
            padding.append([0, 0])
            changes[i] = ['skip', 0, 0]
    if return_changes:
        return np.pad(arr, padding), changes
    else:
        return np.pad(arr, padding)
    
def revert_shape(arr: np.ndarray, changes: dict) -> np.ndarray:
    """
    Compliment to the function `ensure_shape`.
    Takes the input `np.ndarray` arr and `dict` changes to crop / pad the image back to what it was before
    """
    padding = []
    for i in changes.keys():
        if changes[i][0] == "pad":
            padding.append([changes[i][1], changes[i][2]])
        elif changes[i][0] == "crop":
            padding.append([0, 0])
            _slice = tuple([slice(None) for _ in range(i)] + [slice(changes[i][1], -changes[i][2], 1)])
            arr = arr[_slice]
        else:
            padding.append([0, 0])
    
    return np.pad(arr, padding) 


def zero_mean_normalization(arr: np.ndarray, mode: str = "standard", window: Optional[list[float]] = None) -> np.ndarray:
    """
    Function to zero mean normalize the image (i.e. substracting the mean and scaling by the standard deviation).
    Can either do it: 
    * `standard` where the raw mean and standard deviations are used
    * `ignored_zeros` where the mean and standard deviations are calculated while ignoring zeros
    * `windowed` where the mean and standard deviation are calculated between two set values
    ---
    Args:
    * `arr`: `np.ndarray` that is to be normalized
    * `mode`: `str` that defines the normalization type (`standard`, `ignore_zeros`, or `windowed`)
    * `window`: `Optional[list[float]]` that is required if `mode = "windowed"` containing [lower_bound, upper_bound] to be used
    ---
    Returns:
    * `np.ndarray` derived from `arr` that has been normalized accordingly
    """
    if mode == "standard":
        return (arr - arr.mean()) / arr.std()
    
    elif mode == "ignore_zeros":
        mean = arr[arr != 0].mean()
        std = arr[arr != 0].std()
        return (arr - mean) / std
    
    elif mode == "windowed":
        if not window: raise ValueError(f'If "windowed" mode selected, argument "window" must be defined.')
        mean = arr[(arr > window[0]) & (arr < window[1])].mean()
        std = arr[(arr > window[0]) & (arr < window[1])].std()
        return (arr - mean) / std

    else:
        raise ValueError(f'mode: {mode} not found.')

def bounding_box(arr: np.ndarray, option: str = 'cube', scale_to_1: bool = False) -> list[Union[float, int]]:
    """
    Function that finds finds a bounding box based off of a given label.
    Currently, only 3D images are supported.
    ---
    Args:
    * `arr`: `np.ndarray` label that the bounding box is derived from
    * `option`: `str` defining the type of bounding box to return. (`"cube"` currently only supported)
    * `scale_to_1`: `bool` defining whether to return the absolute index or the relative index (between [0,1])
    ---
    Returns:
    * `list[Union[float, int]]` of values depending on the bounding box option:

    Options:
    * `"cube"`: [bottom_x, bottom_y, bottom_z, width (x), height (y), depth (z)]
    """
    if len(arr.shape) != 3: raise ValueError(f'Only 3D volumes are supported, got {len(arr.shape)}')

    # reordering the shape for spacial meshgrid
    x, y, z = arr.shape
    space = [y, x, z]

    # binarizing the label:
    blab = arr > 0

    # Getting the spacial  meshgrid
    dims: list[np.ndarray] = np.meshgrid(*[np.arange(0, s, 1) for s in space])
    
    # Finding the extents of the binarize image
    extents = [[dim[blab].min(), dim[blab].max()] for dim in dims]

    if scale_to_1:
        extents = [[e[0]/s, e[1]/s] for e, s in zip(extents, blab.shape)]

    if option == "cube":
        return [e[0] for e in extents] + [e[1] - e[0] for e in extents]
    else:
        raise ValueError(f'Option {option} not found.')
    
def resample(arr: np.ndarray, src_res: list[float], dst_res: list[float], order:int=0, **kwargs) -> np.ndarray:
    """
    Function that takes an input array and resamples it to the desired resolution.

    ---
    Args:
    * `arr`: `np.ndarray` that is going to be resampled
    * `src_res`: `list[float]` the current resolution of the image
    * `dst_res`: `list[float]` the target resolution of the image
    * `order`: `int` that defines the order of resampling. I.e. the amount of interpolation. ()
    ---
    Returns:
    * `np.ndarray` derived from `arr` that has been resampled to the target resolution.
    """
    if len(arr.shape) != len(src_res) != len(dst_res): raise ValueError(f'Arr shape must match len of src_res and dst_res')

    return ndimg.zoom(arr, [x/y for x, y in zip(src_res, dst_res)], order=order, **kwargs)

def crop_to_bbox(arr: np.ndarray, bbox: list[int], return_pad_amount: bool = False) -> Union[np.ndarray, tuple[np.ndarray, list[list[int]]]]:
    """
    Function that crops an array to a given bounding box.
    The bounding box is currently defined as [x_start, y_start, z_start, width (x), height (y), depth (z)]
    ---
    args:
    * `arr`: `np.ndarray` of the image to be cropped
    * `bbox`: `list[int]` that holds the bounding box as defined above
    * `return_pad_amount`: `bool` that defines if you return the equivalent padding that this cropped off
    ---
    Returns:
    * `np.ndarray` that is a cropped version of `arr`
    - OR -
    * `tuple[np.ndarray, list[list[int]]]` where `np.ndarray` is the same as above and `list[list[int]]` is the padding amount in `np.pad` function.
    """
    # starts = bbox[:3]
    # sizes = bbox[3:]
    # ends = [x + y for x, y in zip(starts, sizes)]

    # THE THINGS DO NOT ALIGN WITH XYZ BUT INSTEAD YXZ
    x, y, z = bbox[:3]
    w, h, d = bbox[3:]
    
    starts = [y, x, z]
    ends = [y+h, x+w, z+d]

    _slice = tuple([slice(s, e, 1) for s, e in zip(starts, ends)])
    out = arr[_slice]
    
    if return_pad_amount:
        pad = [[s, shape - e] for s, e, shape in zip(starts, ends, arr.shape)]
        return out, pad
    else:
        return out
    
def get_largest_blob(mask:np.ndarray, return_all:bool=False) -> np.ndarray:
    """
    Function that takes in a binary mask, finds all connected shapes, and returns the largest object.
    Used to clean up predictions to avoid erronious segmentations. Works best when you expect there to be a single, obvious target.
    ---
    Args:
    * `mask`: `np.ndarray` that holds the mask to the processed
    * `return_all`: `bool` toggle for whether of not to return all the shapes, or just the biggest
    ---
    Out:
    * `np.ndarray` derived from `mask` with either all individal objects as unique values or just the largest depending on `return_all`
    """

    cca = sitk.ConnectedComponentImageFilter()
    cca.SetFullyConnected(False)

    _input = sitk.GetImageFromArray(mask.astype(np.uint8))
    output_ex = cca.Execute(_input)
    output = sitk.GetArrayFromImage(output_ex)

    if return_all: return output

    counts = [0]
    for n in np.unique(output)[1:]:
        counts.append(output[output == n].sum())
    indx = counts.index(max(counts))

    return np.where(output == indx, 1, 0)

