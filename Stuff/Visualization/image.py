import numpy as np
import scipy.ndimage as ndimg
from typing import Optional, Union
import math as m

from .basic import scale_to_one

__all__ = ['crop', 'equal_ratio_pad', 'crop_2d_imgWidth_setHeight', 'get_LeftRight_dim']


def crop(arr: np.ndarray, auxiliary: Optional[list[np.ndarray]] = None, expansion:int=0, threshold:float=0, fine_adjustments: Optional[list[int]] = None, to_one: bool = False):
    if auxiliary is not None and not isinstance(auxiliary, list): auxiliary = [auxiliary]
    
    if expansion: src_arr = ndimg.binary_dilation(arr > 0, iterations=expansion)
    else: src_arr = arr.copy()

    if to_one:
        src_arr = scale_to_one(src_arr)

    n_len = len(arr.shape)
    for n in range(n_len):
        axis = tuple([i for i in range(n_len) if i != n]) # the planes to average across
        slice_bool = np.mean(src_arr.astype(float), axis=axis) > threshold # finding all things that match that threshold along a direction
        if fine_adjustments:
            if fine_adjustments[n] > 0: slice_bool = ndimg.binary_dilation(slice_bool, iterations=fine_adjustments[n])
            elif fine_adjustments[n] < 0: slice_bool = ndimg.binary_erosion(slice_bool, iterations=abs(fine_adjustments[n]))
            else: pass
        crop_slice = tuple([slice(None) if i !=n else slice_bool for i in range(n_len)]) # slicing the volume
        arr = arr[crop_slice] # cropping the volumes
        src_arr = src_arr[crop_slice]
        if auxiliary:
            for i in range(len(auxiliary)):
                auxiliary[i] = auxiliary[i][crop_slice]

    if not auxiliary:
        return arr
    else:
        return arr, *auxiliary

def equal_ratio_pad(images: list[np.ndarray], aux_images: Optional[list[list[np.ndarray]]], return_ratio: bool = False):
    n_imgs = len(images)
    for i in range(n_imgs): assert len(images[i].shape) == 2, f'Images must all be 2D. Image {i} has length {len(images[i].shape)}'
    if aux_images: assert len(aux_images) == n_imgs, f'Auxillary images must have the same number of elements as target images'

    ratios = [img.shape[0] / img.shape[1] for img in images]
    max_ratio = max(ratios)

    equal_images: Union[list[np.ndarray], list[list[np.ndarray]]] = []

    for i in range(n_imgs):
        img = images[i]
        ratio = ratios[i]

        shape = img.shape
        new_x = int(shape[0] / (ratio / max_ratio))
        difference = new_x - shape[0]
        padding = [[m.floor(difference / 2), m.ceil(difference / 2)], [0,0]]

        if not aux_images:
            equal_images.append(np.pad(img, padding))
        else:
            equal_images.append([np.pad(arr, padding) for arr in [img] + aux_images[i]])

    if not return_ratio:
        return equal_images
    else:
        return equal_images, max_ratio


def crop_2d_imgWidth_setHeight(img: np.ndarray, mask: np.ndarray, aux_arrays: list[np.ndarray] = None, threshold: float = 0, y_expansion: Optional[int] = 1, LR_change : Optional[int] = 0):

    # Cropping left / right
    print(img.shape)
    y_bool = np.mean(img, axis=0) > threshold
    if LR_change:
        if LR_change > 0:
            y_bool = ndimg.binary_dilation(y_bool, iterations=int(LR_change))
        else:
            y_bool = ndimg.binary_erosion(y_bool, iterations=int(abs(LR_change)))
    
    img = img[:, y_bool]
    mask = mask[:, y_bool]
    if aux_arrays:
        for i in range(len(aux_arrays)):
            aux_arrays[i] = aux_arrays[i][:, y_bool]

    # Cropping top / down
    print(img.shape)
    x_bool = np.mean(mask, axis=1) > 0
    diff = 2*y_expansion - x_bool.sum()
    x_bool = ndimg.binary_dilation(x_bool, iterations=m.ceil(diff/2))
    img = img[x_bool]
    mask = mask[x_bool]
    if aux_arrays:
        for i in range(len(aux_arrays)):
            aux_arrays[i] = aux_arrays[i][x_bool]

    if aux_arrays:
        return img, mask, *aux_arrays
    else:
        return img, mask


def get_LeftRight_dim(img: np.ndarray, threshold: float = 0, LR_change: Optional[int] = 0):
    y_bool = np.mean(img, axis=0) > threshold
    if LR_change:
        if LR_change > 0:
            y_bool = ndimg.binary_dilation(y_bool, iterations=int(LR_change))
        else:
            y_bool = ndimg.binary_erosion(y_bool, iterations=int(abs(LR_change)))
    img = img[:, y_bool]
    return img.shape
