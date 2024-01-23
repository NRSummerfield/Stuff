import os, glob, pydicom
from typing import Callable, Any

def find_primary_MR_and_CT_dicom_volumes(file: str) -> bool:
    one_dcm = sorted(glob.glob(os.path.join(file, "*.dcm")))[0]
    try:
        header = pydicom.read_file(one_dcm)
        image_type = header[0x0008, 0x0008].value
        modality = header[0x0008, 0x0060].value
        return ('PRIMARY' == image_type[1]) & (modality.upper() in ['MR', 'CT'])
    
    except Exception as e:
        return False
    
def find_non_MR_and_CT_dicom_volumes(file: str) -> bool:
    one_dcm = sorted(glob.glob(os.path.join(file, "*.dcm")))[0]
    try:
        header = pydicom.read_file(one_dcm)
        modality = header[0x0008, 0x0060].value
        return not (modality.upper() in ['MR', "CT"])

    except Exception as e:
        return False

def find_MR_dicom_volumes(file: str) -> bool:
    """
    Function that identifies an "MR" series based off of a standard dicom file name
    """
    volume_parts = os.path.split(file)[-1].split('_')
    cond1 = volume_parts[2].upper() == 'MR'
    return cond1

def dicom_id_finder(file: str) -> str:
    """
    Function that identifies the StudyInstanceUID for a dicom file
    """
    one_dcm = sorted(glob.glob(os.path.join(file, "*.dcm")))[0]
    try:
        header = pydicom.read_file(one_dcm)
        id = header.StudyInstanceUID
        return id
    except Exception as e:
        print(f'Error found while processing {os.path.split(file)[-1]}: {e}')
        return str(0)

def find_RTS_dicom_volumes(file:str):
    """
    Function that identifies an "RTst" series based off of a standard dicom file name
    """
    volume_parts = os.path.split(file)[-1].split('_')
    cond1 = volume_parts[2].upper() == "RTST"
    return cond1

def find_matching_series(
        path:str, 
        num_branches:int, 
        base_searcher: Callable[[str], bool], 
        base_id_finder: Callable[[str], str], 
        dependent_searcher: Callable[[str], bool], 
        dependent_id_finder: Callable[[str], str],
        verbose: bool = False,
        return_relative_paths: bool = False,
        ) -> dict[str, dict[str, Any]]:
    """
    Helper function that can scroll through files and branches to find files that match based off of a header ID.
    Use cases include Dicom volumes and nifit volumes
    ---
    Args:
    * `path`: `str` of the path to the directory that holds the files of interest
    * `num_branches`: `int` value of the number of branching directories. i.e. "/path/patient/series/volumes" would be 2
    * `base_searcher`: `Callable` function that identifies the base volumes
    * `base_id_finder: `Callable` function that identifies the id of each base volume
    * `dependent_searcher`: `Callable` function that identifies the dependent volumes
    * `dependent_id_finder: `Callable` function that identifies the id of each dependent volume
    * `verbose`: `bool` toggle to print out progress reports or not
    * `return_relative_paths`: `bool` toggle to return the absolute or relative path to the files
    ---
    Out:
    * `dict` that holds a sub dict for each patient with keys `{'Source_File': str, 'Dependent_Files': list[str]}`
    """
    paired_volumes = {}
    rrp = len(path)+1 if return_relative_paths else False
    patients = [file for file in sorted(glob.glob(os.path.join(path, "*"))) if os.path.isdir(file)]
    # For each patient
    for p, patient in enumerate(patients):
        if verbose: print(f'Working on {os.path.split(patient)[-1]}... {p+1} / {len(patients)}')
        # Find all files in the branches
        files = sorted(glob.glob(os.path.join(patient, *["*" for _ in range(num_branches)])))
        
        # For each file, find the ones that match the searcher criterior for the base volumes
        base_volumes = [file for file in files if base_searcher(file)]  
        # Finding each ID from the base volumes
        base_volume_ids = [base_id_finder(file) for file in base_volumes]
        # [print(file) for file in base_volumes]

        # For each file, find the ones that match the searcher criterior for the dependent volumes
        dependent_volumes = [file for file in files if dependent_searcher(file)]
        # Finding the references ID from the dependnent volumes
        dependent_volume_reference_ids = [dependent_id_finder(file) for file in dependent_volumes]

        # Looking for matching ids
        matching_files = {}

        # for each recorded base pair
        for base_path, base_id in zip(base_volumes, base_volume_ids):
            # create a sub dict
            matching_files[base_id] = {'Source_File': base_path[rrp:], "Dependent_Files": []}

            # for each found dependent volume pair
            for dep_path, dep_id in zip(dependent_volumes, dependent_volume_reference_ids):
                if base_id == dep_id:
                    matching_files[base_id]["Dependent_Files"].append(dep_path[rrp:])
        
        paired_volumes[os.path.split(patient)[-1]] = matching_files
    
    return paired_volumes


if __name__ == "__main__":
    import json
    from datetime import datetime as dt

    from_MIM = "/Volumes/GlidehurstLab/LabData/RawData/HFHS.ViewRay.RAW"
    assert os.path.exists(from_MIM)

    args = dict(
        path = from_MIM,
        num_branches = 2,
        base_searcher = find_primary_MR_and_CT_dicom_volumes,
        base_id_finder = dicom_id_finder,
        dependent_searcher = find_non_MR_and_CT_dicom_volumes,
        dependent_id_finder = dicom_id_finder,
        verbose=True,
        return_relative_paths=True,
        )

    start = dt.now()
    paired_volumes = find_matching_series(**args)
    print(f'Time taken to find all files: {dt.now() - start}')

    with open(os.path.join(from_MIM, "Matching_MR_Files_withSoda.json"), 'w') as f:
        json.dump(paired_volumes, f, indent=4)

    # # for patient in paired_volumes.keys():
    # #     print(patient)

    # #     for base_id in paired_volumes[patient].keys():
    # #         Source_File = paired_volumes[patient][base_id]['Source_File']
    # #         Dependent_Files = paired_volumes[patient][base_id]['Dependent_Files']
    # #         print(f'\t{os.path.split(Source_File)[-1]}')
    # #         for dep_file in Dependent_Files:
    # #             print(f'\t\t{os.path.split(dep_file)[-1]}')