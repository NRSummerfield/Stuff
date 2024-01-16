import os, glob, json, pickle, shutil

import torch, numpy as np, nibabel as nib
from rt_utils import RTStructBuilder

import BBox_functions as bbfun, BBox_visualization as bbvis
from CoarseSegmentor import CoarseSegmentation, pre_processing_steps_option1
from FineSegmentor import StructureSegmentation
from Stuff.PreProcessing import MakeMask as MM

def saver(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

over_ride = False
coarse_config_path = "Models/CoarseSegmentationConfig.json"
structural_config_path = "Models/StructuralSegmentationConfig.json"
output_for_savedata = "out"
check_point = 'IntermediateSavePoint_2.pkl'
structure_names = "RA LA RV LV AA SVC IVC PA PV LMCA LADA RCA".split(' ')

try:
    with open(check_point, 'rb') as f:
        intermediate_save_point = pickle.load(f)
except:
    intermediate_save_point = {}
 
# ----------------------------------------------------------------
# Getting the dicom files [This is HARD CODED]
# ----------------------------------------------------------------

# if the key "SrcPaths" hasn't been included yet, find paths.
# Optional over_ride argument to do it anyway
# if not intermediate_save_point.get("SrcPaths_done") or over_ride:
if (not intermediate_save_point.get("SrcPaths_done", False)) or over_ride:
    print(f'Getting the src paths...')

    mst_src = "../UWVR_Cardiac_MRsimOnly"
    patients = [file for file in sorted(glob.glob(os.path.join(mst_src, "*"))) if os.path.isdir(file)]

    # The mst_src contains all patients
    SrcPaths = {}
    for patient in patients:
        pid = patient[-3:]

        # Each patient has different "studies"
        studies = [file for file in sorted(glob.glob(os.path.join(patient, '*'))) if os.path.isdir(file)]
        for study in studies:
        
            # Each study has the different dicom
            files = [file for file in sorted(glob.glob(os.path.join(study, "*"))) if os.path.isdir(file)]

            # finding the files of interest
            for file in files:
                
                pieces = os.path.split(file)[-1].split('_')

                # if 2nd pythonic index is "MR" -> Sim
                if pieces[2] == "MR":
                    SrcPaths[f'{pid}.SIM'] = file

                # TODO: define the fractionated volumes (will be done)
                    
    intermediate_save_point['SrcPaths'] = SrcPaths
    intermediate_save_point['SrcPaths_done'] = True
    saver(intermediate_save_point, check_point)
else:
    print(f'Src paths previously identified. Skipping...')

# ----------------------------------------------------------------
# Getting the coarse segmentations based off the paths found above
# ----------------------------------------------------------------
    
print(f'Loading the coarse segmentor for bbox')
with open(coarse_config_path, 'r') as f:
    coarse_config = json.load(f)
    
coarse_segmentor = CoarseSegmentation(
    model = torch.load(os.path.join("Models", coarse_config["ModelPath"])),
    seg_resolution = coarse_config['Resolution'],
    input_shape = coarse_config["InputShape"],
    device = torch.device("cuda:3"),
    pre_processing_steps = lambda x: x.astype(np.float32)
    )

# Try and load "CoarsePreds", if not, create it
try: thing = intermediate_save_point['CoarsePreds']
except: intermediate_save_point['CoarsePreds'] = {}

# if it hasn't been done yet or over_ride, do it
if not intermediate_save_point.get("CoarsePreds_done", False) or over_ride:
    # Try and get all the coarse images. If it failes, quit and save
    try:
        for pid in intermediate_save_point['SrcPaths'].keys():
            # If it hasn't been done yet or over_ride
            if (pid not in intermediate_save_point['CoarsePreds'].keys()) or over_ride:
                print(f'\tRunning pid: {pid}')
                # Get the path
                src_path = intermediate_save_point['SrcPaths'][pid]

                # Get the raw segmentation
                coarse_seg = coarse_segmentor(src_path, return_image=False)
                
                # Getting a single "blob" from the coarse_segmentation:
                single_blob = bbfun.get_largest_blob(coarse_seg)

                # saving to the file
                intermediate_save_point['CoarsePreds'][pid] = single_blob

    # if it breaks, save and quit
    except Exception as e:
        saver(intermediate_save_point, check_point)
        raise AssertionError(f'Coarse Segmentation broken with error: {e}')

    intermediate_save_point["CoarsePreds_done"] = True
    saver(intermediate_save_point, check_point)
else:
    print(f'Coarse Predictions previously done, skipping...')

# ----------------------------------------------------------------
# QAing the coarse segmentation
# ----------------------------------------------------------------
    
# Try to get the QAed bbox dictionary. Otherwise, create a new one
try: thing = intermediate_save_point['QAed_bbox_differences']
except: intermediate_save_point['QAed_bbox_differences'] = {}

# if it hasn't been done yet or over_ride, do it
print(f'QAing the bboxes')
if not (intermediate_save_point.get("QAed_bbox_differences_done", False)) or over_ride:
    # Try and get all the coarse images. If it failes, quit and save
    try:
        for pid in intermediate_save_point['CoarsePreds'].keys():
            # If it hasn't been done yet or over_ride
            if (pid not in intermediate_save_point['QAed_bbox_differences'].keys()) or over_ride:
                print(f'\tRunning pid: {pid}')

                # extracting the bounding box
                bbox = bbfun.bounding_box(arr=intermediate_save_point['CoarsePreds'][pid])
            
                # Getting the background dicm iamge
                image = MM.get_images([intermediate_save_point["SrcPaths"][pid]])[0]
                image = np.moveaxis(image, 0, -1) # matching coords of bbox

                bbox_runner = bbvis.QA_bbox(image, bbox)
                intermediate_save_point['QAed_bbox_differences'][pid] = bbox_runner.get_difference()

    # if it breaks, save and quit
    except Exception as e:
        saver(intermediate_save_point, check_point)
        raise AssertionError(f'Coarse Segmentation broken with error: {e}')
else:
    print(f'BBox QA already done, skipping...')

intermediate_save_point["QAed_bbox_differences_done"] = True
saver(intermediate_save_point, check_point)

# ----------------------------------------------------------------
# Getting the fine segmentation
# ----------------------------------------------------------------


# try: intermediate_save_point.pop("StructuralPreds_done")
# except: ...
# try: intermediate_save_point.pop("StructuralPreds")
# except: ...

print(f'Loading the structural segmentor for labels')
with open(structural_config_path, 'r') as f:
    structural_config = json.load(f)
    
structural_segmentor = StructureSegmentation(
    model = torch.load(os.path.join("Models", structural_config["ModelPath"])),
    seg_resolution = structural_config['Resolution'],
    input_shape = structural_config["InputShape"],
    device = torch.device("cuda:3"),
    pre_processing_steps = pre_processing_steps_option1
    )

# Try and load "CoarsePreds", if not, create it
try: thing = intermediate_save_point['StructuralPreds']
except: intermediate_save_point['StructuralPreds'] = {}

# if it hasn't been done yet or over_ride, do it
if not intermediate_save_point.get("StructuralPreds_done", False) or over_ride:
    # Try and get all the coarse images. If it failes, quit and save
    try:
        for pid in intermediate_save_point['SrcPaths'].keys():
            # If it hasn't been done yet or over_ride
            if (pid not in intermediate_save_point['StructuralPreds'].keys()) or over_ride:
                print(f'\tRunning pid: {pid}')
                # Get the path
                src_path = intermediate_save_point['SrcPaths'][pid]
                coarse_pred = intermediate_save_point['CoarsePreds'][pid]
                bbox_changes = intermediate_save_point['QAed_bbox_differences'][pid]

                # Get the raw segmentation
                structural_seg = structural_segmentor(src_path, coarse_pred, bbox_changes=bbox_changes)
                print(np.unique(structural_seg))
                intermediate_save_point["StructuralPreds"][pid] = structural_seg

    # if it breaks, save and quit
    except Exception as e:
        saver(intermediate_save_point, check_point)
        raise AssertionError(f'Structural Segmentation broken with error: {e}')

    intermediate_save_point["StructuralPreds_done"] = True
    saver(intermediate_save_point, check_point)

else:
    print(f'Structural Predictions previously done, skipping...')

# ----------------------------------------------------------------
# QAing the fine segmentations
# ----------------------------------------------------------------
# try:
#     for pid in intermediate_save_point['SrcPaths'].keys():
#         print(f'\tRunning pid: {pid}')
#         # Get the path
#         src_path = intermediate_save_point['SrcPaths'][pid]
#         img = MM.get_images([src_path])[0]
#         img = np.moveaxis(img, 0, -1)
#         pred = intermediate_save_point["StructuralPreds"][pid]

#         print(np.unique(pred))
#         bbvis.view_pred(img, pred)

# except:
#     ...

# ----------------------------------------------------------------
# Saving the volumes
# ----------------------------------------------------------------
nifti_path = os.path.join(output_for_savedata, "nifti_volumes")
dicom_path = os.path.join(output_for_savedata, "dicoms")
os.makedirs(nifti_path, exist_ok=True)
os.makedirs(dicom_path, exist_ok=True)

for pid in intermediate_save_point['SrcPaths'].keys():
    # getting the information
    pred = intermediate_save_point["StructuralPreds"][pid]
    raw_dcm = intermediate_save_point['SrcPaths'][pid]
    image = MM.get_images([raw_dcm])
    image = np.moveaxis(image, 0, -1)

    # Saving the predictions
    # Save nifti
    nib_img = nib.nifti1.Nifti1Image(image, affine=np.eye(4))
    nib_lab = nib.nifti1.Nifti1Image(pred.astype(np.int16), affine=np.eye(4))
    nib.save(nib_img, os.path.join(nifti_path, f'Image.{pid}.nii.gz'))
    nib.save(nib_lab, os.path.join(nifti_path, f'Prediction.{pid}.nii.gz'))

    # Save dicom
    rtstruct_builder = RTStructBuilder.create_new(dicom_series_path=raw_dcm)
    if structure_names is None: structure_names = [str(i) for i in np.unique(pred)[1:]]

    # moved_pred = np.moveaxis(pred, 0, -1)
    for i in range(1, 12+1):
        rtstruct_builder.add_roi(mask= pred==i, name="Pred_" + structure_names[i-1])

    rtstruct_builder.save(os.path.join(dicom_path, f"Structures.{pid}"))
    if not os.path.exists(os.path.join(dicom_path, f"DicomImage.{pid}.dcms")):
        shutil.copytree(raw_dcm, os.path.join(dicom_path, f"DicomImage.{pid}.dcms"))
