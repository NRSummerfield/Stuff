# Python inherent
import os, glob
from typing import Optional

# Torch
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.backends import cudnn
cudnn.benchmark = True

# Monai
from monai.inferers.utils import sliding_window_inference


from torchmanager_monai import Manager

import numpy as np

from Stuff.PreProcessing import MakeMask as MM
import pydicom as pdcm
import nibabel as nib
import sys
from rt_utils import RTStructBuilder

if sys.version_info >= (3, 9):
    import matplotlib
    matplotlib.use("TkAgg")

import BBox_functions as bbfun
import BBox_visualization as bbvis
import shutil

class FullWorks:
    def __init__(self,
                 dcm_src: str,
                 save_id: str,
                 save_dst: str,

                 coarse_segmentation_model: torch.nn.Module,
                 coarse_segmentation_resolution: list[float],
                 coarse_segmentation_shape: list[int],

                 structure_segmentation_model: torch.nn.Module,
                 structure_segmentation_resolution: list[float],
                 structure_names: Optional[list[str]] = None,
                 device: torch.device = torch.device('cpu')
                 ):
        
        # [Input] Image information
        self.dcm_src = dcm_src
        self.coarse_res = coarse_segmentation_resolution
        self.coarse_shape = coarse_segmentation_shape
        self.seg_res = structure_segmentation_resolution
        self.save_id = save_id
        self.save_dst = save_dst

        # [Input] model / gpu
        self.coarse_model = coarse_segmentation_model
        self.struct_model = structure_segmentation_model
        self.device = device

        # Getting the dcm image
        self.raw_dcm = MM.get_images([self.dcm_src])[0]
        self.raw_dcm = np.moveaxis(self.raw_dcm, 0, -1) # moving z to last index

        # Getting the current resolution
        header = pdcm.read_file(sorted(glob.glob(os.path.join(dcm_src, "*.dcm")))[0])
        self.src_res = [float(n) for n in header[0x0028, 0x0030].value] + [float(header[0x0018, 0x0050].value)]

        # getting the bounding box
        bbox_pred = self.get_bbox()

        # getting the structures
        pred = self.get_structures(bbox_pred)

        # Viewing the prediction
        bbvis.view_pred(self.raw_dcm, pred)

        # Saving the predictions
        # Save nifti
        nib_img = nib.nifti1.Nifti1Image(self.raw_dcm, affine=np.eye(4))
        nib_lab = nib.nifti1.Nifti1Image(pred.astype(np.int16), affine=np.eye(4))
        nib.save(nib_img, os.path.join(self.save_dst, f'Image.{self.save_id}.nii.gz'))
        nib.save(nib_lab, os.path.join(self.save_dst, f'Prediction.{self.save_id}.nii.gz'))
        
        # Save dicom
        rtstruct_builder = RTStructBuilder.create_new(dicom_series_path=self.dcm_src)
        if structure_names is None: structure_names = [str(i) for i in np.unique(pred)[1:]]

        # moved_pred = np.moveaxis(pred, 0, -1)
        for i in range(1, 12+1):
            rtstruct_builder.add_roi(mask= pred==i, name=structure_names[i-1])

        rtstruct_builder.save(os.path.join(self.save_dst, f"Structures.{self.save_id}"))
        if not os.path.exists(os.path.join(self.save_dst, f"DicomImage.{self.save_id}.dcms")):
            shutil.copytree(self.dcm_src, os.path.join(self.save_dst, f"DicomImage.{self.save_id}.dcms"))

    def get_bbox(self):
        img_for_bbox = self.raw_dcm.copy()

        # resampling
        img_for_bbox = bbfun.resample(img_for_bbox, src_res=self.src_res, dst_res=self.coarse_res, order=0)

        # normalization
        img_for_bbox = bbfun.zero_mean_normalization(img_for_bbox, mode="standard")

        # ensuring size as defined
        img_for_bbox, changes = bbfun.ensure_shape(img_for_bbox, self.coarse_shape, return_changes=True) 

        # adding a batch / channel for model
        img_for_bbox = img_for_bbox[None, None]

        # setting as a torch tensor
        img_for_bbox = torch.tensor(img_for_bbox, dtype=torch.float32).to(device)

        # Running the model
        bbox_model = self.coarse_model.to(device)
        bbox_model.eval()
        with torch.no_grad():
            pred = bbox_model(img_for_bbox)

        # Getting a numpy array off of the GPU
        pred = torch.argmax(pred.detach().cpu(), dim=1)[0].numpy()
        img_for_bbox = img_for_bbox.detach().cpu().numpy()[0, 0]

        # Returning the pred to the original volume's shape / resolution
        reverted_pred = bbfun.revert_shape(pred, changes)
        src_res_pred = bbfun.resample(reverted_pred, src_res=self.coarse_res, dst_res=self.src_res, order=0)

        return src_res_pred
    
    def get_structures(self, bbox_pred):
        img_for_seg = self.raw_dcm.copy()

        # Resampling the volume to 1.5 mm^3 for segmentation
        img_for_seg = bbfun.resample(self.raw_dcm, src_res=self.src_res, dst_res=self.seg_res, order=3)
        bb_for_seg = bbfun.resample(bbox_pred, src_res=self.src_res, dst_res=self.seg_res, order=0)

        # Getting the bounding box from the resampled segmentation
        bbox = bbfun.bounding_box(bb_for_seg)

        # Saving a BBox view:
        bbvis.bbox_view(img_for_seg, bb_for_seg, w=[0.15, 0.7, 0.7], ratios=[1, 1, 1], save="RawBBox.pdf")

        # expand everything by some pixels and then QA it
        # ex = 4
        # bbox = np.asarray(bbox) + np.asarray([-ex, -ex, -ex, 2*ex, 2*ex, 2*ex])
        qa_runner = bbvis.QA_bbox(img_for_seg, bbox)
        bbox = qa_runner.bbox.copy()

        # Cropping the image to the bounding box
        cropped_img, pad_amount = bbfun.crop_to_bbox(img_for_seg, bbox=bbox, return_pad_amount=True)

        # normalization
        cropped_img = bbfun.zero_mean_normalization(cropped_img, mode="standard")

        # adding a batch / channel for model
        cropped_img = cropped_img[None, None]

        # setting as a torch tensor
        cropped_img = torch.tensor(cropped_img, dtype=torch.float32).to(device)

        # Running the model
        segmentation_model = self.struct_model.to(device)
        segmentation_model.eval()
        with torch.no_grad():
            pred = sliding_window_inference(cropped_img, roi_size=[96, 96, 96], predictor=segmentation_model, sw_batch_size=1, overlap=0.5)

        # Getting the predicitons from the GPU
        pred = torch.argmax(pred.detach().cpu(), dim=1)[0].numpy()

        # Reverting the prediction back to the original size
        pred = np.pad(pred, pad_amount)
        pred = bbfun.resample(pred, src_res=self.seg_res, dst_res=self.src_res, order=0)
        return pred
    



if __name__ == '__main__':

    dcm_src = "UWVR_Cardiac_MRsimOnly/UWVR_Cardiac_004/2021-04__Studies/UWVR.Cardiac.004_UWVR.Cardiac.004_MR_2021-04-29_140802_VKLZG_50x45_n144__00000" 
    
    coarse_res = [4, 4, 4]
    coarse_shape = [128, 128, 128]
    seg_res = [1.5, 1.5, 1.5]

    bbox_model = "experiments/coarse_segmentation_run2_withFx.exp/best_dice.model"
    segmentation_model = "experiments/UWVR_pseudo_train_run3_alltrain.exp/best_dice.model"

    device = torch.device('cuda:3')

    bbox_model = Manager.from_checkpoint(bbox_model, map_location=torch.device('cpu')).model
    segmentation_model = Manager.from_checkpoint(segmentation_model, map_location=torch.device('cpu')).model

    args = dict(
        dcm_src = dcm_src,
        save_id = '004.SIM',
        save_dst = 'LabeledOutput',
        coarse_segmentation_model = bbox_model,
        coarse_segmentation_resolution = coarse_res,
        coarse_segmentation_shape = coarse_shape,
        structure_segmentation_model = segmentation_model,
        structure_segmentation_resolution = seg_res,
        device = device
        )
    
    FullWorks(**args)


# Loading the raw image
raw_dcm = MM.get_images([dcm_src])[0]
raw_dcm = np.moveaxis(raw_dcm, 0, -1) # moving z to last index

# Getting the current resolution
header = pdcm.read_file(sorted(glob.glob(os.path.join(dcm_src, "*.dcm")))[0])
src_res = [float(n) for n in header[0x0028, 0x0030].value] + [float(header[0x0018, 0x0050].value)]
