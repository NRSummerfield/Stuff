# Python inherent
import os, glob
from typing import Optional, Callable, Union

# Torch
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.backends import cudnn
cudnn.benchmark = True

# Monai
from monai.inferers.utils import sliding_window_inference

import numpy as np

from Stuff.PreProcessing import MakeMask as MM
import pydicom as pdcm
import sys

if sys.version_info >= (3, 9):
    import matplotlib
    matplotlib.use("TkAgg")

import BBox_functions as bbfun

class CoarseSegmentation:
    def __init__(self, 
        model: torch.nn.Module,
        seg_resolution: list[float],
        input_shape: list[int],
        device: torch.device = torch.device('cpu'),
        pre_processing_steps: Callable[[np.ndarray], np.ndarray] = lambda x: x
        ):
        """
        Class that gets a coarse segmentation in the native resolution of the input volume when called
        Used for bounding box purposes
        ---
        Args:
        * `model`: `torch.nn.Module` that is the coarse segmentation model
        * `seg_resolution`: `list[floats]` representing the resolution the segmentation model is expecting
        * `input_shape`: `list[int]` representing the model input shape
        * `pre_processing_steps`: user defined `Callable` object that does the neccesary preprocessing steps (i.e. normalization) for the model input
        """
        self.model = model.to(device)
        self.model.eval()

        self.dst_res = seg_resolution
        self.shape = input_shape
        self.device = device
        self.pps = pre_processing_steps

    def the_works(self, dicom_series_path: str) -> np.ndarray:
        # grabbing the image
        raw_dcm = MM.get_images([dicom_series_path])[0]
        raw_dcm = np.moveaxis(raw_dcm, 0, -1)

        # grabbing the native resoltuion
        header = pdcm.read_file(dicom_series_path)
        src_res = [float(n) for n in header[0x0028, 0x0030].value] + [float(header[0x0018, 0x0050].value)]

        # Resampling to target resolution
        coarse_img = bbfun.resample(raw_dcm, src_res=src_res, dst_res=self.dst_res, order=0)
        # applying any preprocessing
        coarse_img = self.pps(coarse_img)
        # Ensuring the target shape is met
        coarse_img, changes = bbfun.ensure_shape(coarse_img, self.shape, return_changes=True)
        # adding a batch / channel for torch model
        coarse_img = coarse_img[None, None]
        # Converting to a torch tensor
        coarse_img = torch.tensor(coarse_img, dtype=torch.float32).to(self.device)

        # Running the model
        with torch.no_grad():
            coarse_segmentation = self.model(coarse_img)
        
        # Pulling from the GPU into a usable nifti
        coarse_segmentation = torch.argmax(coarse_segmentation.detach().cpu(), dim=1)[0].numpy()
        
        # Returning it to the native resolution of the dicom
        coarse_segmentation = bbfun.revert_shape(coarse_segmentation, changes)
        coarse_segmentation = bbfun.resample(coarse_segmentation, self.dst_res, src_res, order=0)
        return coarse_segmentation
    
    def dicom_series_to_numpy(self, dicom_series_path) -> tuple[np.ndarray, list[float]]:
        # grabbing the image
        raw_dcm = MM.get_images([dicom_series_path])[0]
        raw_dcm = np.moveaxis(raw_dcm, 0, -1)

        # grabbing the native resoltuion
        header = pdcm.read_file(sorted(glob.glob(os.path.join(dicom_series_path, "*.dcm")))[0])
        src_res = [float(n) for n in header[0x0028, 0x0030].value] + [float(header[0x0018, 0x0050].value)]

        return raw_dcm, src_res

    def preprocess_for_prediction(self, image: np.ndarray, src_res: list[float]) -> tuple[torch.Tensor, dict]:
        # Resampling to target resolution
        coarse_img = bbfun.resample(image, src_res=src_res, dst_res=self.dst_res, order=0)
        # applying any preprocessing
        coarse_img = self.pps(coarse_img)
        # Ensuring the target shape is met
        coarse_img, changes = bbfun.ensure_shape(coarse_img, self.shape, return_changes=True)
        # adding a batch / channel for torch model
        coarse_img = coarse_img[None, None]
        # Converting to a torch tensor
        coarse_img = torch.tensor(coarse_img, dtype=torch.float32).to(self.device)
        return coarse_img, changes
    
    def make_prediction(self, image: torch.Tensor) -> np.ndarray:# Running the model
        with torch.no_grad():
            coarse_segmentation = self.model(image)
        
        # Pulling from the GPU into a usable nifti
        coarse_segmentation = torch.argmax(coarse_segmentation.detach().cpu(), dim=1)[0].numpy()
        return coarse_segmentation
    
    def to_native_resolution(self, image: np.ndarray, src_res: list[float], changes: dict, target_shape: Optional[list[int]] = None) -> np.ndarray:
        # Returning it to the native resolution of the dicom
        coarse_segmentation = bbfun.revert_shape(image, changes)
        coarse_segmentation = bbfun.resample(coarse_segmentation, self.dst_res, src_res, order=0)
        if target_shape: coarse_segmentation = bbfun.ensure_shape(coarse_segmentation, target_shape)
        return coarse_segmentation

    def __call__(self, input: str, *, return_image: bool = False) -> Union[np.ndarray, tuple[np.ndarray]]:
        raw_image, src_res = self.dicom_series_to_numpy(input)
        coarse_img, changes = self.preprocess_for_prediction(raw_image, src_res)
        coarse_segmentation = self.make_prediction(coarse_img)
        coarse_segmentation = self.to_native_resolution(coarse_segmentation, src_res, changes, raw_image.shape)
        if return_image:
            return coarse_segmentation, raw_image
        else:
            return coarse_segmentation
        
def pre_processing_steps_option1(arr: np.ndarray) -> np.ndarray:
    out = bbfun.zero_mean_normalization(arr, mode="standard")
    return out

if __name__ == '__main__':
    from torchmanager_monai import Manager
    from datetime import datetime
    
    dcm_src = "UWVR_Cardiac_MRsimOnly/UWVR_Cardiac_004/2021-04__Studies/UWVR.Cardiac.004_UWVR.Cardiac.004_MR_2021-04-29_140802_VKLZG_50x45_n144__00000" 
    
    coarse_res = [4, 4, 4]
    coarse_shape = [128, 128, 128]

    bbox_model = "experiments/coarse_segmentation_run2_withFx.exp/best_dice.model"

    device = torch.device('cuda:3')

    coarse_model = Manager.from_checkpoint(bbox_model, map_location=torch.device('cpu')).model

    thing = CoarseSegmentation(
        model = coarse_model,
        seg_resolution = coarse_res,
        input_shape=[128, 128, 128],
        device=device,
        pre_processing_steps=pre_processing_steps_option1
    )

    start = datetime.now()
    cseg, img = thing(dcm_src, return_image=True)
    end = datetime.now()
    print(f'Run time for 1 prediction: {end - start}')
    print(cseg.shape, img.shape)