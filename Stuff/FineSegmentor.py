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
import BBox_visualization as bbvis

class StructureSegmentation:
    def __init__(self,
        model: torch.nn.Module,
        seg_resolution: list[float],
        input_shape: list[int],
        device: torch.device,
        pre_processing_steps: Callable[[np.ndarray], np.ndarray] = lambda x: x
        ):
        """
        Class that gets a fine segmentation in the native resolution of the input volume when called.
        Requires a prior bounding box to identify the central heart before segmenting
        ---
        Args:
        * `model`: `torch.nn.Module` that is the fine segmentation model
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



    def dicom_series_to_numpy(self, dicom_series_path: str) -> Union[np.ndarray, list[float]]:
        # grabbing the image
        raw_dcm = MM.get_images([dicom_series_path])[0]
        raw_dcm = np.moveaxis(raw_dcm, 0, -1)

        # grabbing the native resoltuion
        header = pdcm.read_file(sorted(glob.glob(os.path.join(dicom_series_path, "*.dcm")))[0])
        src_res = [float(n) for n in header[0x0028, 0x0030].value] + [float(header[0x0018, 0x0050].value)]

        return raw_dcm, src_res
    

    def preprocess_for_prediction(self, raw_image: np.ndarray, src_res: list[float], coarse_prediction: np.ndarray, bbox_changes: Optional[list[int]] = None) -> tuple[torch.Tensor, list[list[int]]]:
        # Getting the BBox
        bbox = bbfun.bounding_box(coarse_prediction, option="cube", scale_to_1=False)
        # Updating BBox if specified
        if bbox_changes: bbox = [b + db for b, db in zip(bbox, bbox_changes)]
        # Cropping to the BBox
        fine_img, pad_amount = bbfun.crop_to_bbox(raw_image, bbox, return_pad_amount=True)
        # Resampling to the target resoluition
        fine_img = bbfun.resample(fine_img, src_res=src_res, dst_res=self.dst_res, order=3)

        # applying any preprocessing
        fine_img = self.pps(fine_img)
         # adding a batch / channel for torch model
        fine_img = fine_img[None, None]
        # Converting to a torch tensor
        fine_img = torch.tensor(fine_img, dtype=torch.float32).to(self.device)

        # # Resampling to target resolution
        # fine_img = bbfun.resample(raw_image, src_res=src_res, dst_res=self.dst_res, order=0)
        # fine_bbox_prior = bbfun.resample(coarse_prediction, src_res=src_res, dst_res=self.dst_res, order=0)
        
        # # Getting the BBox
        # bbox = bbfun.bounding_box(fine_bbox_prior, option="cube", scale_to_1=False) 
        # # Adjusting the bbox if defined
        # if bbox_changes: 
        #     # adjusting for resolution
        #     bbox_changes = [int(n * r) for n, r in zip(bbox_changes, [x/y for x, y in zip(src_res, self.dst_res,)]*2)]
        #     bbox = [b + db for b, db in zip(bbox, bbox_changes)]
        # # cropping to the bbox
        # fine_img, pad_amount = bbfun.crop_to_bbox(fine_img, bbox, return_pad_amount=True)
        # print(fine_img.shape)

        return fine_img, pad_amount
    
    def make_prediction(self, fine_img: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            pred = sliding_window_inference(fine_img, roi_size=self.shape, predictor=self.model, sw_batch_size=1, overlap=0.5)
        # Getting the predicitons from the GPU
        pred = torch.argmax(pred.detach().cpu(), dim=1)[0].numpy()
        return pred


    def to_native_resolution(self, fine_segmentation: np.ndarray, src_res: list[float], pad_amount: list[list[int]]):
        # Reverting the prediction back to the original size
        pred = bbfun.resample(fine_segmentation, src_res=self.dst_res, dst_res=src_res, order=0)
        pred = np.pad(pred, pad_amount)
        return pred

    def __call__(self, input: str, coarse_prediction: np.ndarray, *, bbox_changes: Optional[list[int]] = None, return_image: bool = False) -> Union[np.ndarray, tuple[np.ndarray]]:
        raw_image, src_res = self.dicom_series_to_numpy(input)
        fine_img, pad_amount = self.preprocess_for_prediction(raw_image, src_res, coarse_prediction, bbox_changes)
        fine_segmentation = self.make_prediction(fine_img)
        fine_segmentation = self.to_native_resolution(fine_segmentation, src_res, pad_amount)
        if return_image:
            return fine_segmentation, raw_image
        else:
            return fine_segmentation


if __name__ == '__main__':
    from torchmanager_monai import Manager
    from datetime import datetime
    
    dcm_src = "UWVR_Cardiac_MRsimOnly/UWVR_Cardiac_004/2021-04__Studies/UWVR.Cardiac.004_UWVR.Cardiac.004_MR_2021-04-29_140802_VKLZG_50x45_n144__00000" 
    
    coarse_res = [4, 4, 4]
    coarse_shape = [128, 128, 128]
    seg_shape = [96, 96, 96]
    seg_res = [1.5, 1.5, 1.5]


    bbox_model = "experiments/coarse_segmentation_run2_withFx.exp/best_dice.model"
    segmentation_model = "experiments/UWVR_pseudo_train_run3_alltrain.exp/best_dice.model"

    device = torch.device('cuda:3')

    bbox_model = Manager.from_checkpoint(bbox_model, map_location=torch.device('cpu')).model
    segmentation_model = Manager.from_checkpoint(segmentation_model, map_location=torch.device('cpu')).model


    from CoarseSegmentor import CoarseSegmentation, pre_processing_steps_option1
    coarse_runner = CoarseSegmentation(
        model = bbox_model,
        seg_resolution = coarse_res,
        input_shape=coarse_shape,
        device=device,
        pre_processing_steps=pre_processing_steps_option1
    )

    fine_runner = StructureSegmentation(
        model= segmentation_model,
        seg_resolution= seg_res,
        input_shape= seg_shape,
        device= device,
        pre_processing_steps= pre_processing_steps_option1,
    )

    cseg_start = datetime.now()
    cseg = coarse_runner(dcm_src)
    print(f'Time for coarse segmentation: {datetime.now() - cseg_start}')


    sseg_start = datetime.now()
    sseg = fine_runner(dcm_src, cseg, bbox_changes=[-3, -3, -3, 6, 6, 6])
    print(f'Time for structural segmentation: {datetime.now() - sseg_start}')

    print(sseg.shape, np.unique(sseg))

