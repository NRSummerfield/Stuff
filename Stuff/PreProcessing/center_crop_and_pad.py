import numpy as np, nibabel as nib, matplotlib.pyplot as plt, scipy.ndimage as ndimg
import os, glob, math as m
from typing import Union, Optional

def center_crop_and_pad(arr: np.ndarray, mask: np.ndarray, target_shape: Union[list[int], int], aux_arr: Optional[Union[np.ndarray, list[np.ndarray]]] = None) -> Union[tuple[np.ndarray], tuple[np.ndarray, list[np.ndarray]]]:
    """
    ## Takes and array and crops it based off of a given mask. A new image with the target shape will be made by cropping and zero-padding around the center of mass.

    ---
    Args:
    * arr: `np.ndarray` input that is cropped based off of the mask
    * mask: `np.ndarray` input that defines the cropping
    * target_shape: `Union[list[int], int]` that defines the target shape
    * aux_arr: `Optional[Union[np.ndarray, list[np.ndarray]]]` that are auxiliary arrays to be cropped and padded in the same fasion
    ---
    Returns:
    * tuple[np.ndarray]
    0 = crop and padded input `arr`
    1 = crop and padded input `mask`
    2 = [Optional] auxiliary arrays
    """
    # QA inputs
    if isinstance(target_shape, int): target_shape = [target_shape for _ in arr.shape]
    assert len(target_shape) == len(arr.shape) == len(mask.shape), f'arguments `target_shape`, `arr`, and `mask` must have the same length of dimensions. Got {len(target_shape)}, {len(arr.shape)}, and {len(mask.shape)}.'

    # Calculate the center of mass to base the coordinate system around
    center_of_mass = [int(x) for x in ndimg.center_of_mass(np.where(mask > 0, 1, 0))]

    # calculate the left / right size of the shape (accounting for odd numbers)
    half_shape = [s / 2 for s in target_shape]
    HS_floor = [m.floor(s) for s in half_shape]
    HS_ceil = [m.ceil(s) for s in half_shape]

    # Calculate the bounding indexs for the target shape over the center of mass
    high_bound = [com + ceil for com, ceil in zip(center_of_mass, HS_ceil)]
    low_bound = [com - floor for com, floor in zip(center_of_mass, HS_floor)]

    # examine the bounding indexes to decide if the image needs to be cropped or zero-padded - HIGH bound
    difference_high_bound_vs_real = [_high_bound - _shape  for _shape, _high_bound in zip(arr.shape, high_bound)]
    # if positive, high bound is GREATER than the image shape and the image needs to be zero-padded
    high_padding = [diff if diff > 0 else 0 for diff in difference_high_bound_vs_real]
    # if negative, high bound is LOWER than the image shape and the image needs to be cropped
    high_cropping = [diff if diff < 0 else _shape for diff, _shape in zip(difference_high_bound_vs_real, arr.shape)]

    # examine the boudnign indexes to decide if the image needs to be cropped or zero-padded - LOW bound
    difference_low_bound_vs_real = [_low_bound - 0 for _low_bound in low_bound]
    # if positive, the low_bound is GREATER than 0 (bottom), the image needs to be cropped
    low_cropping = [diff if diff > 0 else 0 for diff in difference_low_bound_vs_real]
    # if negative, the low bound is LOWER than 0 (bottom), the image needs to be padded
    low_padding = [abs(diff) if diff < 0 else 0 for diff in difference_low_bound_vs_real]

    # cropping and padding the image according to the above
    cropping = tuple([slice(low, high) for low, high in zip(low_cropping, high_cropping)])
    padding = [[low, high] for low, high in zip(low_padding, high_padding)]
    arr = np.pad(arr[cropping], padding)
    mask = np.pad(mask[cropping], padding)
    
    if not isinstance(aux_arr, (list, tuple)): aux_arr = [aux_arr]
    if aux_arr[0] is not None:
        aux_cropped = [np.pad(aux[cropping], padding) for aux in aux_arr]
        return arr, mask, aux_cropped
    else: return arr, mask


if __name__ == '__main__':

    n = 0
    src = "/mnt/data/Summerfield/Data/Select_CCTA_withFengSubstructures/1.5mm_cubic_21StructModel/Volumes"
    image_path = sorted(glob.glob(os.path.join(src, 'Image.0*.SIM.nii.gz')))[n]
    label_path = sorted(glob.glob(os.path.join(src, 'Label.0*.SIM.nii.gz')))[n]

    image = nib.load(image_path).get_fdata()[0]
    label = nib.load(label_path).get_fdata()[0]

    aux_sens = [None, image, [image, image]]
    for aux_sen in aux_sens:
        if aux_sen is not None:
            img, lab, aux = center_crop_and_pad(image, label, 128, aux_sen)
            print(img.shape, lab.shape)
            [print(_aux.shape) for _aux in aux]
        else:
            img, lab = center_crop_and_pad(image, label, 128, aux_sen)
            print(img.shape, lab.shape)
